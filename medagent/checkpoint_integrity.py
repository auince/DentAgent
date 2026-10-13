"""Select a complete training checkpoint without loading tensors into memory."""
import json
import struct
import zipfile
from pathlib import Path


def _safetensors_complete(path):
    size=path.stat().st_size
    with path.open('rb') as stream:
        raw=stream.read(8)
        if len(raw)!=8:raise ValueError('missing_safetensors_header:'+path.name)
        length=struct.unpack('<Q',raw)[0]
        if length<=0 or length>100_000_000 or length+8>size:
            raise ValueError('invalid_safetensors_header:'+path.name)
        header=json.loads(stream.read(length))
    offsets=[]
    for name,tensor in header.items():
        if name=='__metadata__':continue
        start,end=tensor['data_offsets']
        if not 0<=start<=end<=size-length-8:
            raise ValueError('incomplete_tensor_payload:'+path.name)
        offsets.append((start,end))
    if not offsets or max(end for _,end in offsets)!=size-length-8:
        raise ValueError('incomplete_safetensors_payload:'+path.name)


def _weights_complete(folder):
    weights=list(folder.glob('*.safetensors'))
    for index in folder.glob('*.safetensors.index.json'):
        shards=set(json.loads(index.read_text())['weight_map'].values())
        if not shards:raise ValueError('empty_weight_index')
        for name in shards:
            path=folder/name
            if path.parent!=folder or not path.exists():raise ValueError('missing_weight_shard:'+name)
            if path not in weights:weights.append(path)
    if not weights:raise ValueError('missing_weights:'+folder.name)
    for path in weights:_safetensors_complete(path)
    if (folder/'adapter_model.safetensors').exists():
        config=json.loads((folder/'adapter_config.json').read_text())
        if not isinstance(config,dict) or not config.get('peft_type'):raise ValueError('invalid_adapter_config')


def _torch_archive_complete(path):
    # A partial torch.save normally lacks its ZIP central directory. Check it
    # without unpickling or making another full optimizer copy in memory.
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if not entries or not any(item.filename.endswith('/data.pkl') for item in entries):
            raise ValueError('incomplete_torch_archive:'+path.name)


def checkpoint_issue(folder,world_size=2,require_reference_adapter=False):
    folder=Path(folder)
    try:
        step=int(folder.name.split('-')[-1])
        state=json.loads((folder/'trainer_state.json').read_text())
        if state.get('global_step')!=step:raise ValueError('trainer_step_mismatch')
        _weights_complete(folder)
        for name in ('optimizer.pt','scheduler.pt'):_torch_archive_complete(folder/name)
        rng=['rng_state.pth'] if world_size==1 else [f'rng_state_{rank}.pth' for rank in range(world_size)]
        for name in rng:_torch_archive_complete(folder/name)
        if require_reference_adapter:_weights_complete(folder/'ref')
        return None
    except (OSError,ValueError,KeyError,TypeError,struct.error,zipfile.BadZipFile) as exc:
        return str(exc)[:240]


def select_checkpoint(folder,world_size=2,require_reference_adapter=False):
    candidates=sorted(Path(folder).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
    rejected=[]
    for candidate in candidates:
        issue=checkpoint_issue(candidate,world_size,require_reference_adapter)
        if issue is None:return candidate,rejected
        rejected.append({'checkpoint':candidate.name,'reason':issue})
    if candidates:raise ValueError('no_complete_checkpoint_available:'+json.dumps(rejected))
    return None,rejected

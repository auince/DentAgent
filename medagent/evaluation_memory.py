"""Bound the logits peak for text-only Agentic RL validation."""


def memory_bounded_grpo_trainer(base):
    class MemoryBoundedGRPOTrainer(base):
        def _full_logits_logps(
            self, model, input_ids, attention_mask, logits_to_keep,
            batch_size=None, compute_entropy=False, compute_aux_loss=False,
            **kwargs,
        ):
            import torch

            parent = super()._full_logits_logps
            common = dict(compute_entropy=compute_entropy,
                          compute_aux_loss=compute_aux_loss, **kwargs)
            # Training and multimodal scoring keep the upstream implementation.
            # The active agent passes text tokens for external tool observations.
            token_fields = {'token_type_ids', 'mm_token_type_ids'}
            multimodal = any(value is not None for name, value in kwargs.items()
                             if name not in token_fields)
            if self.model.training or torch.is_grad_enabled() or multimodal:
                return parent(model, input_ids, attention_mask, logits_to_keep,
                              batch_size=batch_size, **common)

            # Each parent invocation returns before the next forward starts,
            # releasing full logits and their fp32 conversion between rows.
            # Reassemble row scores before upstream group advantages and loss.
            scores = []
            for index in range(input_ids.size(0)):
                row_kwargs = dict(common)
                for name in token_fields:
                    value = row_kwargs.get(name)
                    if value is not None:
                        row_kwargs[name] = value[index:index + 1]
                scores.append(parent(
                    model, input_ids[index:index + 1],
                    attention_mask[index:index + 1], logits_to_keep,
                    batch_size=1, **row_kwargs,
                ))
            logps = torch.cat([score[0] for score in scores], dim=0)
            entropies = (torch.cat([score[1] for score in scores], dim=0)
                         if compute_entropy else None)
            aux_loss = (torch.stack([score[2] for score in scores]).mean()
                        if compute_aux_loss else None)
            return logps, entropies, aux_loss

    return MemoryBoundedGRPOTrainer

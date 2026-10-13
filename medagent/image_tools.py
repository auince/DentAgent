"""In-memory, allowlisted image tools; no arbitrary paths or shell commands."""
import base64
import io
import json
from .common import digest, parse_json


class ImageTools:
    def __init__(self, client, images):
        self.client, self.images, self.events, self.ocr_results = client, {}, [], {}
        if len(images) > 8:
            raise ValueError('too_many_images')
        for i, image in enumerate(images):
            self.register('image_'+str(i), image)

    def register(self, key, data):
        from PIL import Image, ImageOps
        if not isinstance(data,str) or not data.startswith(('data:image/png;base64,','data:image/jpeg;base64,','data:image/webp;base64,')):
            raise ValueError('invalid_image_format')
        if len(data) > 16_000_000 or len(self.images) >= 16:
            raise ValueError('image_budget_exceeded')
        raw = base64.b64decode(data.split(',',1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in {'PNG','JPEG','WEBP'} or im.width*im.height > 16_000_000:
                raise ValueError('image_dimensions_exceeded')
            image = ImageOps.exif_transpose(im).convert('RGB')
            self.images[key] = image.copy()
        return key

    def data_url(self, key):
        out = io.BytesIO()
        self.images[key].save(out, format='PNG')
        data = 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
        if len(data)>16_000_000:
            raise ValueError('encoded_image_too_large')
        return data

    def inventory(self):
        return [{'id':k,'width':im.width,'height':im.height} for k,im in self.images.items()]

    def call(self, action):
        if not isinstance(action,dict) or action.get('tool') not in {'crop','zoom','ocr'}:
            raise ValueError('unknown_image_tool')
        if len(self.events) >= 6:
            raise ValueError('image_tool_budget_exhausted')
        key=action.get('image_id')
        if not isinstance(key,str) or key not in self.images:
            raise ValueError('unknown_image_id')
        im=self.images[key]; tool=action['tool']; new='derived_'+str(len(self.events))
        if tool=='crop':
            box=action.get('box')
            if not isinstance(box,list) or len(box)!=4 or any(type(x)is not int for x in box):
                raise ValueError('crop_box_must_be_four_pixel_integers')
            x1,y1,x2,y2=box
            if not (0<=x1<x2<=im.width and 0<=y1<y2<=im.height):
                raise ValueError('crop_box_outside_image')
            self.images[new]=im.crop(tuple(box)); result={'image_id':new,'box':box,'parent':key}
        elif tool=='zoom':
            from PIL import Image
            scale=action.get('scale')
            if type(scale) not in (int,float) or not 1<scale<=4:
                raise ValueError('zoom_scale_outside_budget')
            size=(round(im.width*scale),round(im.height*scale))
            if size[0]*size[1]>16_000_000:
                raise ValueError('zoom_dimensions_exceeded')
            self.images[new]=im.resize(size,Image.Resampling.LANCZOS)
            result={'image_id':new,'scale':scale,'parent':key}
        else:
            raw=parse_json(self.client.complete('vision','逐字OCR转写，保留牙位、否定、数字、单位；模糊字符标记[无法辨认]。只输出 {"text":"原文","issues":[]}。',
                '转写指定图像区域，不补写检查。',[self.data_url(key)]))
            if not isinstance(raw,dict) or not isinstance(raw.get('text'),str) or not isinstance(raw.get('issues'),list):
                raise ValueError('invalid_ocr_schema')
            oid='ocr_'+str(len(self.events)); result={'ocr_id':oid,'image_id':key,**raw}
            self.ocr_results[oid]=result
        self.events.append({'tool':tool,'input_image_id':key,'result':result})
        return result

    def initial_text(self):
        texts=[];issues=[]
        # Initial OCR is necessary only if the user supplied no text.
        for key in list(self.images):
            raw=parse_json(self.client.complete('vision','逐字转写，输出 {"text":"原文","issues":[]}；不清楚处写[无法辨认]，不得补字。','病历图像',[self.data_url(key)]))
            if not isinstance(raw,dict) or not isinstance(raw.get('text'),str) or not isinstance(raw.get('issues'),list):
                raise ValueError('invalid_ocr_schema')
            texts.append(raw['text']);issues+=raw['issues']
        if not any(t.strip() for t in texts):raise ValueError('empty_ocr')
        return '\n'.join(texts),issues

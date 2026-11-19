"""Counted transformations of immutable 10ms sources into native16 variants."""
from pathlib import Path
import hashlib
import re
ROOT=Path(__file__).resolve().parents[1]
BASE_IDENTITIES = {'ns_routes': {'bytes': 44952, 'sha256': 'a7cacbd822f22b006133e94e88a300905d0c1decca1e514b3efca61d0692b710'}, 'ns_streaming': {'bytes': 19023, 'sha256': 'ed3a83361d12d6d3076888e3273006e0ad893a7592059f9140100c770cd5c2f3'}, 'ns_stream_worker': {'bytes': 20894, 'sha256': '36fe16a98d08d2d45e2f36ec68ea84e8ddd9bed975ebeab687ac1c1e18453fcb'}, 'aec_routes': {'bytes': 35079, 'sha256': 'c815f9452fcc0d9791ca95436229212b2dc050e6cc0243aecc8eb1438183c2cd'}}

def load_base(name, namespace, *, replace=(), patterns=()):
    path=ROOT/f"server/{name}.py"
    lock=BASE_IDENTITIES[name]
    data=path.read_bytes()[:lock['bytes']]
    if hashlib.sha256(data).hexdigest()!=lock['sha256']:
        raise RuntimeError(f"native16 base source changed: {name}")
    text=data.decode()
    guard='if __name__ == "__main__":'
    if guard in text:text=text[:text.rindex(guard)]
    for old,new,count in replace:
        actual=text.count(old)
        if actual!=count:raise RuntimeError(f"native16 replacement count mismatch {name}: {old!r} {actual}!={count}")
        text=text.replace(old,new)
    for pattern,new,count in patterns:
        text,actual=re.subn(pattern,new,text)
        if actual!=count:raise RuntimeError(f"native16 regex count mismatch {name}: {pattern!r} {actual}!={count}")
    exec(compile(text,str(path)+":native16",'exec'),namespace)

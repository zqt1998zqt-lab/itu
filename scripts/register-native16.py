"""Append isolated native16 dispatch without modifying any original prefix."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
MARKER = '# Native16 registration: additive dispatch; original source prefix remains frozen.'


def main():
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError('disk safety threshold; preserve files and stop')
    appendices = {
        'server/ns_routes.py': '''
_native16_original_ns_backend_for = ns_backend_for
_native16_original_resolve_ns_cli = resolve_ns_cli
_native16_original_ns_route_identity = ns_route_identity
_native16_original_validate_ns_route_report = validate_ns_route_report
_native16_original_execute_ns_command = execute_ns_command

def _is_native16_configuration(configuration):
    return isinstance(configuration, dict) and isinstance(configuration.get("selection"), dict) and configuration["selection"].get("ns") == "cofilite_native16"

def ns_backend_for(configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import ns_backend_for as native
        return native(configuration)
    return _native16_original_ns_backend_for(configuration)

def resolve_ns_cli(default_cli, configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import resolve_ns_cli as native
        return native(default_cli, configuration)
    return _native16_original_resolve_ns_cli(default_cli, configuration)

def ns_route_identity(configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import ns_route_identity as native
        return native(configuration)
    return _native16_original_ns_route_identity(configuration)

def validate_ns_route_report(report, configuration):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import validate_ns_route_report as native
        return native(report, configuration)
    return _native16_original_validate_ns_route_report(report, configuration)

def execute_ns_command(command, *, configuration, timeout=300, **kwargs):
    if _is_native16_configuration(configuration):
        from server.ns_routes_native16 import execute_ns_command as native
        return native(command, configuration=configuration, timeout=timeout, **kwargs)
    return _native16_original_execute_ns_command(command, configuration=configuration, timeout=timeout, **kwargs)
''',
        'server/aec_routes.py': '''
_native16_original_admit_aec_route = admit_aec_route
_native16_original_read_aec_capabilities = read_aec_capabilities
_native16_original_validate_aec_route_report = validate_aec_route_report
_native16_original_validate_admitted_aec_identity = validate_admitted_aec_identity

def _is_native16_configuration(configuration):
    return isinstance(configuration, dict) and isinstance(configuration.get("selection"), dict) and configuration["selection"].get("ns") == "cofilite_native16"

def _is_native16_cli(cli):
    return Path(cli).resolve() == (Path(__file__).resolve().parents[1] / "build/native16-20261117/host/cpp/itu_audio_cli").resolve()

def admit_aec_route(cli, configuration, **kwargs):
    if _is_native16_configuration(configuration):
        from server.aec_routes_native16 import admit_aec_route as native
        return native(cli, configuration, **kwargs)
    if _is_native16_cli(cli):
        raise ValueError("原生16 ms CLI 只能用于明确选择的 native16 算法簇")
    return _native16_original_admit_aec_route(cli, configuration, **kwargs)

def read_aec_capabilities(cli):
    if _is_native16_cli(cli):
        from server.aec_routes_native16 import read_aec_capabilities as native
        return native(cli)
    return _native16_original_read_aec_capabilities(cli)

def validate_aec_route_report(report, configuration):
    if _is_native16_configuration(configuration):
        from server.aec_routes_native16 import validate_aec_route_report as native
        return native(report, configuration)
    return _native16_original_validate_aec_route_report(report, configuration)

def validate_admitted_aec_identity(report, capability):
    if isinstance(capability, dict) and capability.get("id") == "localvqe_aec200k" and capability.get("delay_samples") == 256:
        from server.aec_routes_native16 import validate_admitted_aec_identity as native
        return native(report, capability)
    return _native16_original_validate_admitted_aec_identity(report, capability)
''',
        'server/synthesize.py': '''
CHAIN_CATALOG = ROOT / "src/features/enhancement-chain-configs-native16.json"
''',
        'server/alignment_runner.py': '''
CHAIN_CATALOG = PROJECT_ROOT / "src/features/enhancement-chain-configs-native16.json"

@router.get("/aec-backends-native16")
def get_aec_backends_native16():
    cli = PROJECT_ROOT / "build/native16-20261117/host/cpp/itu_audio_cli"
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise HTTPException(503, "原生16 ms VCP CLI 尚未构建")
    try:
        result = read_aec_capabilities(cli)
    except ValueError as error:
        raise HTTPException(503, str(error)) from error
    return {**result, "cli_sha256": sha256_file(cli), "frame_samples": 256, "callback_ms": 16}
''',
    }
    changes = {}
    for relative, appendix in appendices.items():
        path = ROOT / relative
        data = path.read_bytes()
        if MARKER.encode() in data:
            raise RuntimeError(f'native16 registration already present, do not duplicate: {path}')
        changes[relative] = {'prefix_bytes': len(data), 'prefix_sha256': hashlib.sha256(data).hexdigest()}
        with path.open('a') as stream:
            stream.write('\n\n' + MARKER + '\n' + appendix)
    out = ROOT / 'build/native16-20261117/registration.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(changes, stream, indent=2)
    print(json.dumps(changes))


if __name__ == '__main__':
    main()

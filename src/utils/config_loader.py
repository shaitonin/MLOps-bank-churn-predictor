import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """carrega os yamls
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
    
#import yaml → biblioteca para ler YAML
#Path → manipular caminhos de arquivos de forma moderna
#Any → permite qualquer tipo de valor no dicionário
#path.exists() → verifica se o arquivo existe
#yaml.safe_load() → converte YAML em dict
#or {} → evita retornar None (garante um dicionário vazio)

#A função load_yaml(path: Path):
#    * Recebe o caminho de um arquivo (Path)
#    * Verifica se o arquivo existe
#    * Abre o arquivo YAML
#    * Converte o conteúdo em um dicionário Python
#    * Retorna esse dicionário
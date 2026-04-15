from transformers import AutoTokenizer, BertModel, RobertaModel
import os
from pathlib import Path


def _default_bert_path():
    # Prefer explicit override if set
    env_path = os.environ.get("BERT_BASE_UNCASED_PATH")
    if env_path:
        return env_path

    # Try to locate repo root from this file, then use <repo>/third_party/bert-base-uncased
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "third_party":
            return str(parent / "bert-base-uncased")

    # Fallback to CWD-based guess
    return str(Path.cwd() / "third_party" / "bert-base-uncased")


def get_tokenlizer(text_encoder_type, bert_base_uncased_path=None):
    """仅使用本地模型的分词器加载函数"""
    
    if text_encoder_type == "bert-base-uncased":
        local_path = bert_base_uncased_path or _default_bert_path()
        
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"本地BERT模型未找到: {local_path}\n"
                f"请下载BERT模型到该路径，或使用以下命令下载:\n"
                f"python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained('{local_path}')\""
            )
        
        print(f"使用本地BERT分词器: {local_path}")
        return AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    
    # 如果不是bert-base-uncased，就抛出错误
    raise ValueError(f"仅支持本地bert-base-uncased模型，当前请求: {text_encoder_type}")


def get_pretrained_language_model(text_encoder_type, bert_base_uncased_path=None):
    """仅使用本地模型的语言模型加载函数"""
    
    if text_encoder_type == "bert-base-uncased":
        local_path = bert_base_uncased_path or _default_bert_path()
        
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"本地BERT模型未找到: {local_path}\n"
                f"请下载BERT模型到该路径，或使用以下命令下载:\n"
                f"python -c \"from transformers import BertModel; "
                f"BertModel.from_pretrained('bert-base-uncased').save_pretrained('{local_path}')\""
            )
        
        print(f"使用本地BERT模型: {local_path}")
        return BertModel.from_pretrained(local_path, local_files_only=True)
    
    # 如果不是bert-base-uncased，就抛出错误
    raise ValueError(f"仅支持本地bert-base-uncased模型，当前请求: {text_encoder_type}")

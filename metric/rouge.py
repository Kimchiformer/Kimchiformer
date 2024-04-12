from typing import *
import evaluate
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch import Tensor
from torch.utils.data import DataLoader


class RougeEvaluator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel
    ) -> None:
        self.evaluator = evaluate.load("rouge")
        self.tokenizer = tokenizer
        self.model = model

    def __call__(
        self,
        dataset: DataLoader,
        loader: Callable[[Dict], Tuple[List[str], List[str]]]
    ) -> Dict[str, float]:
        tgt_full = []
        pred_full = []

        for batch in dataset:
            src, tgt = loader(batch)
            pred = self._generate(src)
            tgt_full += tgt
            pred_full += pred

        return self.evaluator.compute(
            predictions=pred_full,
            references=tgt_full
        )

    def _generate(self, text: List[str]) -> List[str]:
        enc = self._encode(text)
        gen = self.model.generate(**enc)
        dec = self._decode(gen)
        return dec

    def _encode(self, text: List[str]) -> Dict[str, Tensor]:
        return self.tokenizer(text, padding=True, return_tensors="pt")

    def _decode(self, tensor: Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
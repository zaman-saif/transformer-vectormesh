"""Text vectorization components using HuggingFace models."""

import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer

from vectormesh.types import Cachable


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class BaseVectorizer(ABC, Cachable):
    """
    Base class for all vectorizers.

    All vectorizers must:
    - Have a model_name, device, and metadata
    - model_name is used in the VectorCache to identify the model used for a vectorizer
    - col_name is used to store the output of the vectorizer in the dataset
    - device is for hardware acceleration, if applicable
    - Implement __call__ that returns dict[str, list[Float[Tensor, "..."]]]
    - The exact tensor dimensionality can vary by implementation
    """

    model_name: str
    col_name: str
    device: str = Field(default_factory=detect_device)

    _metadata: Any = PrivateAttr()
    _effective_max_length: Optional[int] = PrivateAttr(default=None)

    @abstractmethod
    @model_validator(mode="after")
    def initialize_model(self):
        """
        Initialize the model/API connection.
        Must set self._metadata with at least:
        - hidden_size or dim: output dimension
        Must set self._effective_max_length to the actual context limit used,
        or None if the concept does not apply (e.g. RegexVectorizer).
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, texts: list[str], batchsize: int
    ) -> dict[str, list[Float[Tensor, "..."]]]:
        """
        Process texts and return embedding.

        Args:
            texts: List of input texts
            batchsize: Batch size for processing

        Returns:
            Dict with '{self.col_name : list[Tensor]}'.
            Tensor dimensionality varies by implementation
        """
        pass

    @property
    def get_metadata(self) -> dict:
        """
        Return metadata about the model.
        Subclasses can override to add more fields.
        """
        return {
            "model_name": self.model_name,
            "col_name": self.col_name,
            "hidden_size": getattr(self._metadata, "hidden_size"),
            "context_size": self._effective_max_length,
        }

    @property
    def get_hidden_size(self) -> int:
        return getattr(self._metadata, "hidden_size")

    @property
    def get_context_size(self) -> Optional[int]:
        return self._effective_max_length


class Vectorizer(BaseVectorizer):
    model_name: str
    col_name: str
    device: str = Field(default_factory=detect_device)
    max_length: Optional[int] = None

    _metadata: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _stride: int = PrivateAttr()
    _effective_max_length: int = PrivateAttr()
    chunk_sizes: Counter = Counter()

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = AutoConfig.from_pretrained(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        max_pos = getattr(self._metadata, "max_position_embeddings")
        self._effective_max_length = (
            min(max_pos, self.max_length) if self.max_length else max_pos
        )
        self._stride = self._effective_max_length // 10

        logger.info(f"Using device: {self.device}")
        logger.info(
            f"using max_length: {self._effective_max_length} (model max: {max_pos}), stride: {self._stride}"
        )
        return self

    @jaxtyped(typechecker=beartype)
    def tokenize(
        self, text: list[str]
    ) -> tuple[
        Int[Tensor, "batch tokens"],
        Int[Tensor, "batch tokens"],
        Int[Tensor, "batch"],
    ]:
        """
        Receives a batch of texts Σ* , where Σ is an alphabet, and * represents that the strings
        can be concatenated in any order to create a sequence to create sentences.

        The output is:
        - input_ids: A 2D tensor of token ids (batch_size * chunks, max_length) ∈ ℕ
        - attention_mask: A 2D tensor of attention mask (batch_size * chunks, max_length) ∈ {0, 1}
        - overflow_to_sample_mapping: A 1D tensor of document indices (batch_size * chunks,) ∈ ℕ  (eg 0, 0 ,0, 1, 1, 2, ...)

        Because of the context window (eg 512 tokens), we "overflow" the tokens into
        a (batch * chunks, max_length) tensor.
        eg batch might be 32, but some documents are 3 * 512 tokens, others are 5 * 512 tokens, etc.
        So we end up with eg (115, 512) tokens from an input of (32) documents

        We will later reconstruct into
        (chunks, max_length) for each document with the help of the overflow indices
        """
        tokens = self._tokenizer(
            text,
            truncation=True,
            max_length=self._effective_max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = tokens["input_ids"]
        attention = tokens["attention_mask"]
        overflow = tokens["overflow_to_sample_mapping"]
        return input_ids, attention, overflow

    @jaxtyped(typechecker=beartype)
    def embed(
        self,
        input_ids: Int[Tensor, "batch tokens"],
        attention: Int[Tensor, "batch tokens"],
        batchsize: int,
    ) -> tuple[Float[Tensor, "batch tokens dim"], Int[Tensor, "batch tokens"]]:
        """
        This function turns a 2D tensor (batch * chunks, tokens) ∈ ℕ  into an embedding
        (batch * chunks, tokens, dim) ∈ ℝ

        The attention mask is used to mask out padding tokens.
        batchsize is the number of chunks to be processed at once
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention.to(self.device)
            chunks = input_ids.shape[0]
            embs = []
            for i in range(0, chunks, batchsize):
                input_ids_batch = input_ids[i : i + batchsize]
                attention_mask_batch = attention_mask[i : i + batchsize]
                outputs = self._model(
                    input_ids_batch, attention_mask=attention_mask_batch
                )
                embs.append(outputs.last_hidden_state)
        embeddings = torch.cat(embs, dim=0)

        return embeddings, attention_mask

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        embeddings: Float[Tensor, "batch tokens dim"],
        attention: Int[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch dim"]:
        """
        This function turns a 3D tensor (batch, tokens, dim) ∈ ℝ
        into an embedding (batch, dim) ∈ ℝ by aggregating over the tokens dimension.

        We can do this because due to the attention mechanism, all
        tokens have been "mixed" like a hologram and
        sort-of contain the information of the full contextwindow.
        """
        mask_expand = attention.unsqueeze(-1)
        sum_emb = torch.sum(embeddings * mask_expand, dim=1)
        sum_mask = torch.sum(mask_expand, dim=1)
        return sum_emb / sum_mask

    @jaxtyped(typechecker=beartype)
    def extend(
        self,
        agg: Float[Tensor, "batch dim"],
        overflow: Int[Tensor, "batch"],
        num_docs: int,
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        """
        With the help of the overflow indices, we can regroup the embeddings back into
        a (chunks, dim) ∈ ℝ tensor per document where chunk varies per document.
        """
        regrouped = []
        for doc_idx in range(num_docs):
            idx = overflow == doc_idx
            embed = agg[idx]
            self.chunk_sizes[embed.shape[0]] += 1
            regrouped.append(embed)
        return {self.col_name: regrouped}

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, texts: list[str], batchsize: int
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        input_ids, attention, overflow = self.tokenize(texts)
        embedded, attention = self.embed(input_ids, attention, batchsize=batchsize)
        agg = self.aggregate(embedded, attention)
        return self.extend(agg, overflow, num_docs=len(texts))

    @property
    def get_model(self):
        return self._model

    @property
    def get_tokenizer(self):
        return self._tokenizer


class RegexVectorizer(BaseVectorizer):
    """
    Vectorizer that creates binary feature vectors based on regex pattern matches.
    """

    model_name: str = "regex_vectorizer"
    col_name: str = "regex_features"
    training_texts: Optional[list[str]] = Field(
        default=None, description="Texts to fit on during initialization"
    )
    min_doc_frequency: int = Field(
        default=50, description="Minimum documents a pattern must appear in"
    )
    max_features: int = Field(
        default=1000, description="Maximum number of features (top-k patterns)"
    )
    pattern_builder: Callable[[], re.Pattern] = Field(
        description="Function that returns compiled regex pattern"
    )
    harmonizer: Callable[[tuple], str] = Field(
        description="Function that harmonizes match groups into canonical form"
    )

    _pattern_to_idx: dict[str, int] = PrivateAttr()
    _compiled_pattern: re.Pattern = PrivateAttr()
    _match_counts: Optional[Counter] = PrivateAttr(default=None)
    _doc_frequencies: Optional[Counter] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def initialize_model(self):
        """
        Initialize with compiled pattern.
        """
        self._pattern_to_idx = {}
        self._compiled_pattern = self.pattern_builder()
        self._match_counts = None
        self._doc_frequencies = None

        class RegexMetadata:
            def __init__(self, max_features: int):
                self.hidden_size = max_features

        self._metadata = RegexMetadata(self.max_features)
        self._effective_max_length = None

        if self.training_texts is not None:
            self.fit(self.training_texts)
        return self

    def _compute_match_counts(self, texts: list[str]) -> tuple[Counter, Counter]:
        match_counts = Counter()
        doc_frequencies = Counter()

        for text in texts:
            matches = self._compiled_pattern.findall(text)
            doc_patterns = set()

            for match in matches:
                harmonized = self.harmonizer(match)
                match_counts[harmonized] += 1
                doc_patterns.add(harmonized)

            for pattern in doc_patterns:
                doc_frequencies[pattern] += 1

        return match_counts, doc_frequencies

    def fit(self, texts: list[str]) -> "RegexVectorizer":
        # Compute or reuse cached counts
        if self._match_counts is None or self._doc_frequencies is None:
            self._match_counts, self._doc_frequencies = self._compute_match_counts(
                texts
            )

        # Filter by minimum document frequency
        filtered_patterns = [
            pattern
            for pattern, doc_freq in self._doc_frequencies.items()
            if doc_freq >= self.min_doc_frequency
        ]

        # Take top-k by total frequency
        top_patterns = [
            pattern
            for pattern, _ in self._match_counts.most_common(self.max_features)
            if pattern in filtered_patterns
        ]

        # Store pattern lookup
        self._pattern_to_idx = {
            pattern: i for i, pattern in enumerate(top_patterns[: self.max_features])
        }

        # Update metadata with actual feature count
        self._metadata.hidden_size = len(self._pattern_to_idx)

        logger.info(f"Fitted {len(self._pattern_to_idx)} patterns")

        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, texts: list[str], batchsize: int = 32
    ) -> dict[str, list[Float[Tensor, "hidden_size"]]]:
        if not self._pattern_to_idx:
            raise RuntimeError(
                "Vectorizer must be fitted before calling. Run .fit(texts) first."
            )

        vectors = []

        for text in texts:
            # Create zero vector
            vector = torch.zeros(len(self._pattern_to_idx), dtype=torch.float32)

            # Find all matches
            matches = self._compiled_pattern.findall(text)
            doc_patterns = set()

            for match in matches:
                harmonized = self.harmonizer(match)
                doc_patterns.add(harmonized)

            # Set binary features
            for pattern in doc_patterns:
                if pattern in self._pattern_to_idx:
                    idx = self._pattern_to_idx[pattern]
                    vector[idx] = 1.0

            vectors.append(vector.to(self.device))

        return {self.col_name: vectors}

    def print_stats(
        self, texts: Optional[list[str]] = None, top_k: int = 20, plot: bool = True
    ):
        if texts is None:
            if self._match_counts is None:
                raise RuntimeError(
                    "No cached match counts. Either call fit() first or provide texts."
                )
            match_counts = self._match_counts
        else:
            match_counts, _ = self._compute_match_counts(texts)

        logger.info(f"Total unique patterns: {len(match_counts)}")
        logger.info(f"most commonn {match_counts.most_common(top_k)}")

        if plot and len(match_counts) > 0:
            plt.figure(figsize=(12, 6))
            top_matches = match_counts.most_common(min(50, len(match_counts)))
            labels = [ref for ref, _ in top_matches]
            counts = [count for _, count in top_matches]

            plt.bar(range(len(counts)), counts)
            plt.xticks(range(len(labels)), labels, rotation=90, ha="right")
            plt.xlabel("Pattern")
            plt.ylabel("Frequency")
            plt.title(f"Top {len(counts)} Pattern Matches")
            plt.tight_layout()
            plt.show()

    @property
    def get_metadata(self) -> dict:
        """Return metadata about the vectorizer"""
        base_metadata = super().get_metadata
        base_metadata.update(
            {
                "min_doc_frequency": self.min_doc_frequency,
                "max_features": self.max_features,
            }
        )
        return base_metadata


def build_legal_reference_pattern() -> re.Pattern:
    """Build the regex pattern for Dutch legal references
    eg:
        "artikel 265 Boek 3 van het Burgerlijk Wetboek",
        "artikel 7:2 Burgerlijk Wetboek",
        "artikel 7:26 lid 3 van het Burgerlijk Wetboek",
        "artikelen 6:251 en 6:252 Burgerlijk Wetboek",
        "artikel 55 Wet Bodembescherming"
    """
    article_prefix = "[Aa]rt(?:ikel(?:en)?)?"
    article_number = r"\d+(?::\d+)?"
    article_modifier = r"(?:\s+(?:en\s+\d+(?::\d+)?|lid\s+\d+))?"
    book_reference = r"(?:\s+[Bb]oek\s+\d+)?"
    connector = r"(?:\s+van\s+het\s+)?"

    # law_name = (
    #    r"(?:[Bb]urgerlijk\s+[Ww]etboek|\bBW\b|[Ww]et\s+[Bb]odembescherming|\bWbb\b)|[Vv]erordening|[Ww]et"
    # )

    laws = [
        r"[Bb]urgerlijk\s+[Ww]etboek", r"\bBW\b",
        r"[Ww]etboek\s+van\s+[Bb]urgerlijke\s+[Rr]echtsvordering", r"\bRv\b",
        r"[Ww]etboek\s+van\s+[Kk]oophandel", r"\bWvK\b",
        r"[Ww]etboek\s+van\s+[Ss]trafrecht", r"\bSr\b",
        r"[Ww]etboek\s+van\s+[Ss]trafvordering", r"\bSv\b",
        r"[Aa]lgemene\s+[Ww]et\s+[Bb]estuursrecht", r"\bAwb\b",
        r"[Ww]et\s+[Aa]lgemene\s+[Bb]epalingen\s+[Oo]mgevingsrecht", r"\bWabo\b",
        r"[Ww]et\s+[Rr]uimtelijke\s+[Oo]rdening", r"\bWro\b",
        r"[Gg]rondwet", r"\bGw\b",
        r"[Ee]uropees\s+[Vv]erdrag\s+[Rr]echten\s+[Vv]an\s+[Dd]e\s+[Mm]ens", r"\bEVRM\b",
        r"[Hh]andvest\s+[Gg]rondrechten\s+[Ee]uropese\s+[Uu]nie",
        r"[Aa]lgemene\s+[Ww]et\s+[Ii]nzake\s+[Rr]ijksbelastingen", r"\bAWR\b",
        r"[Ww]et\s+[Ii]nkomstenbelasting",
        r"[Ww]et\s+[Bb]odembescherming", r"\bWbb\b",
        r"[Ww]et\s+[Mm]ilieubeheer", r"\bWm\b",
        r"[Vv]erordening",
        r"[Ww]et",
        r"[Ee]uropees",
        r"[Hh]andvest",
        r"[Aa]lgemene",
        r"[Bb]esluit",
        r"[Vv]erordening",
    ]
    law_name = "(?:" + "|".join(sorted(laws, key=len, reverse=True)) + ")"

    full_pattern = (
        rf"\b{article_prefix}\s+"
        rf"({article_number}{article_modifier}{book_reference})\s+"
        rf"{connector}({law_name})"
    )
    return re.compile(full_pattern)


def harmonize_legal_reference(match: tuple) -> str:
    """Convert legal reference match to harmonized format"""
    article_ref, law_name = match
    law_lower = law_name.lower()

    if "burgerlijk wetboek" in law_lower or law_lower == "bw":
        law_abbr = "BW"
    elif "bodembescherming" in law_lower or law_lower == "wbb":
        law_abbr = "Bodem"
    else:
        law_abbr = law_name

    return f"{article_ref} {law_abbr}"


def build_imdb_review_pattern() -> re.Pattern:
    """Match film-related vocabulary in IMDB reviews.

    Each match is a single word/phrase, e.g. "great", "terrible", "acting",
    "horror". After fitting, each unique matched term becomes one binary
    feature dimension (1 if the word appears anywhere in the text, 0 if not).

    Examples:
        "The acting was great but the plot was terrible"
        → matches: ["acting", "great", "plot", "terrible"]
    """
    sentiment = (
        "great|good|bad|poor|excellent|terrible|awful|brilliant|boring|"
        "amazing|weak|superb|dreadful|outstanding|mediocre|impressive|"
        "masterpiece|hilarious|touching|predictable|formulaic"
    )
    genre = (
        "horror|comedy|thriller|drama|romance|action|western|"
        "documentary|musical|mystery|fantasy|animation"
    )
    craft = (
        "acting|performance|screenplay|script|direction|plot|"
        "cinematography|dialogue|soundtrack|editing"
    )
    return re.compile(rf"\b(?:{sentiment}|{genre}|{craft})\b", re.IGNORECASE)


def harmonize_imdb_match(match: str) -> str:
    """Normalize a matched film vocabulary word to lowercase."""
    return match.lower()

"""
NeuroCore Dil ve Çeviri Katmanı (L6)
=====================================
Dil algılama, çok sağlayıcılı çeviri, terim sözlüğü ve API köprüsünü
dışa aktarır.

Hızlı başlangıç:
    from neurocore.language import LanguageLayer

    layer = LanguageLayer()
    layer.initialize(config, event_bus)
    print(layer.detect("Merhaba dünya"))
    print(layer.translate("yapay sinir ağı", "en"))
    layer.shutdown()
"""

from neurocore.language.api_bridge import ApiBridge, ApiProvider, mask_key
from neurocore.language.detector import (
    LanguageDetector,
    LanguageGuess,
    guess_encoding,
    is_probably_code,
    script_of,
)
from neurocore.language.dictionary import (
    TerminologyEntry,
    TerminologyStore,
    normalize_language_code,
)
from neurocore.language.layer import LanguageLayer
from neurocore.language.translator import (
    ApiTranslator,
    BaseTranslator,
    CascadeTranslator,
    DictionaryTranslator,
    IdentityTranslator,
    NeuralTranslator,
    TranslationCache,
    TranslationResult,
)

__all__ = [
    # Katman cephesi
    "LanguageLayer",
    # Algılama
    "LanguageDetector",
    "LanguageGuess",
    "guess_encoding",
    "is_probably_code",
    "script_of",
    # Sözlük
    "TerminologyStore",
    "TerminologyEntry",
    "normalize_language_code",
    # Çeviri
    "BaseTranslator",
    "IdentityTranslator",
    "DictionaryTranslator",
    "NeuralTranslator",
    "ApiTranslator",
    "CascadeTranslator",
    "TranslationCache",
    "TranslationResult",
    # API köprüsü
    "ApiBridge",
    "ApiProvider",
    "mask_key",
]

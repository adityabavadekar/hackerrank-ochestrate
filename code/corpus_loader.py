from typing import List, Dict
from pathlib import Path

import yaml

from . import config
from .logger import get_logger
from .models import Document, Company, SourceType

logger = get_logger(__name__)


class CorpusLoader:

    def load(self) -> List[Document]:
        documents: List[Document] = []
        logger.info("Loading corpus from %s", config.CORPUS_DIR)

        for source_dir in config.CORPUS_DIR.iterdir():
            if not source_dir.is_dir():
                continue

            try:
                source_type = self._map_source(source_dir.name)
            except ValueError:
                logger.warning("Skipping unknown corpus directory: %s", source_dir.name)
                continue

            for file_path in source_dir.rglob("*.md"):
                if str(file_path).endswith("index.md"):
                    # skip index files
                    continue
                doc = self._load_file(file_path, source_type)
                if doc:
                    documents.append(doc)

        logger.info("Corpus loaded: %d total documents", len(documents))
        return documents

    def _load_file(self, path: Path, source: SourceType) -> Document:
        try:
            raw = path.read_text(encoding="utf-8")
            meta, body = self._extract_frontmatter(raw)
            body = self._clean_content(body)
            
            if not self._is_meaningful(body):
                return None
            
            GENERIC = {"support", "general", "index"}

            # Extract product_area from directory structure
            rel_path = path.relative_to(config.CORPUS_DIR)
            
            if len(rel_path.parts) >= 3:
                area = rel_path.parts[-2].lower().replace("-", "_")
                if area in GENERIC and len(rel_path.parts) >= 4:
                    area = rel_path.parts[-3].lower().replace("-", "_")
                meta["product_area"] = area
            else:
                meta["product_area"] = "general"

            return Document(
                id=self._build_id(path),
                content=self._enrich_content(path, body, meta),
                source=self._map_company(source),
                meta=meta
            )

        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            return None

    def _build_id(self, path: Path) -> str:
        return str(path.relative_to(config.CORPUS_DIR))
        
    def _map_source(self, name: str) -> SourceType:
        name = name.lower()
        if name == "claude":
            return SourceType.CLAUDE
        if name == "hackerrank":
            return SourceType.HACKERRANK
        if name == "visa":
            return SourceType.VISA

        raise ValueError(f"Unknown source: {name}")
    
    def _extract_frontmatter(self, content: str) -> tuple[dict, str]:
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)

        if len(parts) < 3:
            return {}, content

        _, fm, body = parts

        try:
            meta = yaml.safe_load(fm) or {}
        except Exception:
            meta = {}

        return meta, body.strip()

    def _is_meaningful(self, body: str) -> bool:
        """Filter to skip effectively empty documents that just contain titles/dates."""
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("_Last modified:") or line.startswith("_Last updated:"):
                continue
            if line == "---":
                continue
            return True  # Found at least one meaningful line
        return False
        
    def _map_company(self, source: SourceType) -> Company:
        mapping = {
            SourceType.CLAUDE: Company.CLAUDE,
            SourceType.HACKERRANK: Company.HACKERRANK,
            SourceType.VISA: Company.VISA
        }
        return mapping[source]
    
    def _enrich_content(self, path: Path, content: str, meta: dict) -> str:
        parts = path.parts

        # Normalize path to remove leading dirs before "data"
        try:
            data_idx = parts.index("data")
            hierarchy = " > ".join(parts[data_idx + 1:-1])
        except ValueError:
            hierarchy = " > ".join(parts[:-1])

        title = meta.get("title", path.stem.replace("-", " ").title())

        # include breadcrumb/category if available
        breadcrumbs = meta.get("breadcrumbs") or []
        if breadcrumbs:
            hierarchy = " > ".join(breadcrumbs)

        return (
            f"TITLE: {title}\n"
            f"CATEGORY: {hierarchy}\n\n"
            f"{content}"
        )

    def _clean_content(self, body: str) -> str:
        import re
        from markdownify import markdownify
        
        # 1. Remove long sequences of hyphens, asterisks, equals signs
        body = re.sub(r'(?m)^[-=_*]{10,}\s*$', '', body)
        
        # 2. Remove all HTML image tags
        body = re.sub(r'<img\b[^>]*>', '', body)
        
        # 3. Remove all markdown images: ![alt](url)
        body = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', body)
        
        # 4. Convert HTML tables and other HTML to markdown
        body = markdownify(body, strip=['img'], bs4_options={"features": "lxml"})
        
        # 5. Clean up multiple blank lines
        body = re.sub(r'\n{3,}', '\n\n', body)
        
        return body.strip()
    
    def grouped_by_company(self, documents: List[Document]) -> Dict[Company, List[Document]]:
        grouped = {}
        for doc in documents:
            company = doc.source
            if company not in grouped:
                grouped[company] = []
            grouped[company].append(doc)
        return grouped

if __name__ == "__main__":
    from .logger import get_logger as _gl
    _log = _gl("corpus_loader.__main__")

    loader = CorpusLoader()
    documents = loader.load()
    grouped = loader.grouped_by_company(documents)

    _log.info("Corpus loading summary:")
    for company, docs in grouped.items():
        _log.info("-> %-12s  %d documents", company.value, len(docs))
    _log.info("Total: %d documents", len(documents))

    _log.info("Sample document per company:")
    for company, docs in grouped.items():
        if not docs:
            continue

        print("===" * 20, company.value, "===" * 20)
        for i, sample in enumerate(docs[:2]):
            print(f"\n--- SAMPLE {i+1} ---")
            print("ID:\n",sample.id)
            print("\nMETA:\n", sample.meta)
            print("\nCONTENT:\n", sample.content[:-1] + "...")
            print("\n")
        print()
        print()

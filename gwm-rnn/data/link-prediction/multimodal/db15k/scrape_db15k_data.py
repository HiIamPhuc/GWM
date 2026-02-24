"""
DB15K Data Scraping Script

This script collects textual and visual information for DB15K entities from:
1. Triple files (train.txt, valid.txt, test.txt)
2. URI parsing with semantic relation mappings
3. DBpedia SPARQL queries (optional, for text descriptions)
4. DBpedia image queries and downloads (optional)

Output:
- Processed triples (numeric IDs)
- Entity and relation vocabularies
- Entity and relation text descriptions
- Downloaded images (cached)
- Metadata

Usage:
    python scrape_db15k_data.py --help
"""

import argparse
import json
import time
import re
from pathlib import Path
from tqdm.auto import tqdm
import requests
from io import BytesIO
from PIL import Image


# ============================================================================
# SEMANTIC RELATION MAPPINGS
# ============================================================================

RELATION_MAPPINGS = {
    # RDF/RDFS core
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 'is a type of',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is subclass of',
    'http://www.w3.org/2000/01/rdf-schema#subPropertyOf': 'is subproperty of',
    'http://www.w3.org/2000/01/rdf-schema#domain': 'has domain',
    'http://www.w3.org/2000/01/rdf-schema#range': 'has range',
    'http://www.w3.org/2000/01/rdf-schema#label': 'has label',
    'http://www.w3.org/2000/01/rdf-schema#comment': 'has comment',
    'http://www.w3.org/2000/01/rdf-schema#seeAlso': 'see also',
    
    # OWL core
    'http://www.w3.org/2002/07/owl#sameAs': 'is same as',
    'http://www.w3.org/2002/07/owl#equivalentClass': 'is equivalent class to',
    'http://www.w3.org/2002/07/owl#equivalentProperty': 'is equivalent property to',
    'http://www.w3.org/2002/07/owl#inverseOf': 'is inverse of',
    'http://www.w3.org/2002/07/owl#differentFrom': 'is different from',
    
    # Common properties
    'http://xmlns.com/foaf/0.1/knows': 'knows',
    'http://xmlns.com/foaf/0.1/name': 'has name',
    'http://purl.org/dc/terms/creator': 'created by',
    'http://purl.org/dc/terms/created': 'created on',
    'http://purl.org/dc/terms/modified': 'modified on',
}


# ============================================================================
# TRIPLE PARSING
# ============================================================================

def parse_triples(file_path):
    """Parse triples."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    triples = []
    for line in lines:
        parts = line.strip(' .').split(' ')
        if len(parts) != 3:
            continue
        h, r, t = parts
        triples.append((h, r, t))
    
    return triples


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_uri(uri):
    """
    Extract readable text from DBpedia URI.
    
    Examples:
        <http://dbpedia.org/resource/Albert_Einstein> → "Albert Einstein"
        <http://dbpedia.org/ontology/birthPlace> → "birth place"
    """
    uri_clean = uri.strip('<>')
    
    # Check if it's a known ontology relation
    if uri_clean in RELATION_MAPPINGS:
        return RELATION_MAPPINGS[uri_clean]
    
    # Get the last part after the last slash or hash
    if '#' in uri_clean:
        text = uri_clean.split('#')[-1]
    else:
        text = uri_clean.split('/')[-1]
    
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    
    # Handle camelCase: insert space before capitals
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If still empty or very short, try to extract from path
    if len(text) < 2 and '/' in uri_clean:
        parts = uri_clean.split('/')
        if len(parts) >= 2:
            text = ' '.join(parts[-2:])
            text = text.replace('_', ' ')
    
    return text if text else uri_clean


def extract_text_from_dbpedia_sparql(uri, timeout=10):
    """Query DBpedia SPARQL endpoint for entity label and description."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        print("⚠️  SPARQLWrapper not installed. Install with: pip install sparqlwrapper")
        return extract_text_from_uri(uri)
    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setTimeout(timeout)
    
    clean_uri = uri.strip('<>')
    
    query = f"""
    SELECT ?label ?description WHERE {{
        <{clean_uri}> rdfs:label ?label .
        OPTIONAL {{ <{clean_uri}> dbo:description ?description . }}
        FILTER (lang(?label) = 'en')
        FILTER (!bound(?description) || lang(?description) = 'en')
    }}
    LIMIT 1
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        time.sleep(0.5)  # Rate limiting
        results = sparql.query().convert()
        
        if results and "results" in results and "bindings" in results["results"]:
            bindings = results["results"]["bindings"]
            
            if bindings:
                result = bindings[0]
                label = result.get("label", {}).get("value", "")
                description = result.get("description", {}).get("value", "")
                
                if description:
                    text = f"{label}. {description}"
                else:
                    text = label
                
                if text:
                    return text
    except Exception:
        pass
    
    # Fallback to URI extraction
    return extract_text_from_uri(uri)


# ============================================================================
# IMAGE DOWNLOADING
# ============================================================================

def query_dbpedia_for_image(uri, timeout=10):
    """Query DBpedia SPARQL endpoint for entity image URL."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        return None
    
    clean_uri = uri.strip('<>')
    
    query = f"""
    SELECT ?image WHERE {{
        {{
            <{clean_uri}> dbo:thumbnail ?image .
        }} UNION {{
            <{clean_uri}> foaf:depiction ?image .
        }}
    }}
    LIMIT 1
    """
    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setTimeout(timeout)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        time.sleep(0.5)  # Rate limiting
        results = sparql.query().convert()
        
        if results and "results" in results and "bindings" in results["results"]:
            bindings = results["results"]["bindings"]
            if bindings and "image" in bindings[0]:
                return bindings[0]["image"]["value"]
    except Exception:
        pass
    
    return None


def download_and_save_image(image_url, save_path, max_size_mb=5):
    """Download image from URL and save to disk."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        response = requests.get(image_url, timeout=10, stream=True, verify=True, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return False
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            return False
        
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(save_path, 'JPEG', quality=85)
        
        return True
    except Exception:
        return False


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Scrape DB15K data from triple files and DBpedia')
    
    # Input/Output paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw DB15K data directory (containing train.txt, valid.txt, test.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    
    # Text extraction options
    parser.add_argument('--text_mode', type=str, default='uri_only',
                        choices=['uri_only', 'dbpedia_sparql'],
                        help='Text extraction method: uri_only (fast) or dbpedia_sparql (slow, detailed)')
    parser.add_argument('--text_sample', type=int, default=None,
                        help='Sample only N entities for DBpedia text queries (for testing)')
    
    # Image downloading options
    parser.add_argument('--download_images', action='store_true',
                        help='Download images from DBpedia')
    parser.add_argument('--image_sample', type=int, default=50,
                        help='Sample only N entities for image downloads (default: 50)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to download (None = all)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DB15K DATA SCRAPING")
    print("="*70)
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Text mode: {args.text_mode}")
    print(f"Download images: {args.download_images}")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load triples
    # ========================================================================
    print("\n[1/5] Loading triples...")
    train_triples = parse_triples(data_dir / 'train.txt')
    valid_triples = parse_triples(data_dir / 'valid.txt')
    test_triples = parse_triples(data_dir / 'test.txt')
    
    print(f"  Train: {len(train_triples):,}")
    print(f"  Valid: {len(valid_triples):,}")
    print(f"  Test: {len(test_triples):,}")
    
    # ========================================================================
    # STEP 2: Create vocabularies
    # ========================================================================
    print("\n[2/5] Creating vocabularies...")
    entities_set = set()
    relations_set = set()
    
    for h, r, t in train_triples + valid_triples + test_triples:
        entities_set.add(h)
        entities_set.add(t)
        relations_set.add(r)
    
    entities = sorted(list(entities_set))
    relations = sorted(list(relations_set))
    
    entity2id = {ent: idx for idx, ent in enumerate(entities)}
    relation2id = {rel: idx for idx, rel in enumerate(relations)}
    
    print(f"  Entities: {len(entities):,}")
    print(f"  Relations: {len(relations):,}")
    
    # ========================================================================
    # STEP 3: Extract entity and relation texts
    # ========================================================================
    print("\n[3/5] Extracting texts...")
    
    entity_texts = []
    if args.text_mode == 'uri_only':
        print("  Method: URI parsing (fast)")
        for entity_uri in tqdm(entities, desc="  Entities"):
            text = extract_text_from_uri(entity_uri)
            entity_texts.append(text)
    else:
        print("  Method: DBpedia SPARQL (slow)")
        sample_size = args.text_sample if args.text_sample else len(entities)
        print(f"  Processing {sample_size:,} entities...")
        
        for i, entity_uri in enumerate(tqdm(entities[:sample_size], desc="  Entities")):
            text = extract_text_from_dbpedia_sparql(entity_uri)
            entity_texts.append(text)
        
        # Use URI extraction for remaining
        for entity_uri in entities[sample_size:]:
            text = extract_text_from_uri(entity_uri)
            entity_texts.append(text)
    
    relation_texts = []
    for relation_uri in tqdm(relations, desc="  Relations"):
        text = extract_text_from_uri(relation_uri)
        relation_texts.append(text)
    
    print(f"  ✓ Extracted {len(entity_texts):,} entity texts")
    print(f"  ✓ Extracted {len(relation_texts):,} relation texts")
    
    # ========================================================================
    # STEP 4: Download images (optional)
    # ========================================================================
    image_info = {}
    
    if args.download_images:
        print("\n[4/5] Downloading images from DBpedia...")
        
        image_cache_dir = output_dir / 'images'
        image_cache_dir.mkdir(exist_ok=True)
        
        num_to_process = min(args.image_sample, len(entities))
        if args.max_images:
            num_to_process = min(num_to_process, args.max_images)
        
        print(f"  Processing {num_to_process:,} entities...")
        
        success_count = 0
        for entity_id in tqdm(range(num_to_process), desc="  Querying & downloading"):
            entity_uri = entities[entity_id]
            image_path = image_cache_dir / f"{entity_id}.jpg"
            
            # Check if already cached
            if image_path.exists():
                image_info[entity_id] = str(image_path.relative_to(output_dir))
                success_count += 1
                continue
            
            # Query and download
            image_url = query_dbpedia_for_image(entity_uri)
            if image_url:
                if download_and_save_image(image_url, image_path):
                    image_info[entity_id] = str(image_path.relative_to(output_dir))
                    success_count += 1
        
        print(f"  ✓ Downloaded {success_count:,} images")
        print(f"  Coverage: {success_count/num_to_process*100:.1f}%")
    else:
        print("\n[4/5] Skipping image downloads (use --download_images to enable)")
    
    # ========================================================================
    # STEP 5: Save everything
    # ========================================================================
    print("\n[5/5] Saving data...")
    
    # Save triples (raw URIs)
    with open(output_dir / 'triples_train.txt', 'w', encoding='utf-8') as f:
        for h, r, t in train_triples:
            f.write(f"{h} {r} {t}\n")
    
    with open(output_dir / 'triples_valid.txt', 'w', encoding='utf-8') as f:
        for h, r, t in valid_triples:
            f.write(f"{h} {r} {t}\n")
    
    with open(output_dir / 'triples_test.txt', 'w', encoding='utf-8') as f:
        for h, r, t in test_triples:
            f.write(f"{h} {r} {t}\n")
    
    # Save vocabularies
    with open(output_dir / 'entity2id.json', 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'relation2id.json', 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, indent=2, ensure_ascii=False)
    
    # Save texts
    with open(output_dir / 'entity_texts.json', 'w', encoding='utf-8') as f:
        json.dump(entity_texts, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'relation_texts.json', 'w', encoding='utf-8') as f:
        json.dump(relation_texts, f, indent=2, ensure_ascii=False)
    
    # Save image info
    with open(output_dir / 'image_paths.json', 'w', encoding='utf-8') as f:
        json.dump(image_info, f, indent=2)
    
    # Save metadata
    metadata = {
        'dataset': 'DB15K',
        'num_entities': len(entities),
        'num_relations': len(relations),
        'num_train_triples': len(train_triples),
        'num_valid_triples': len(valid_triples),
        'num_test_triples': len(test_triples),
        'text_extraction_mode': args.text_mode,
        'images_downloaded': args.download_images,
        'num_images': len(image_info),
        'image_coverage': len(image_info) / len(entities) if entities else 0.0,
    }
    
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved to {output_dir}")
    
    print("\n" + "="*70)
    print("✅ DATA SCRAPING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  triples_train.txt, triples_valid.txt, triples_test.txt")
    print("  entity2id.json, relation2id.json")
    print("  entity_texts.json, relation_texts.json")
    print("  image_paths.json")
    if args.download_images:
        print(f"  images/ ({len(image_info)} images)")
    print("  metadata.json")
    print("\nNext step: Run prepare_embeddings.ipynb to generate embeddings")


if __name__ == '__main__':
    main()

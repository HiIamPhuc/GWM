"""
MKG (Wikidata) Data Scraping Script

This script collects textual and visual information for MKG-W/MKG-Y entities from:
1. Triple files (train.txt, valid.txt, test.txt)
2. URI parsing with Wikidata entity/property IDs
3. Wikidata SPARQL queries (optional, for labels and descriptions)
4. Wikidata image queries and downloads (optional)

Output:
- Processed triples (numeric IDs)
- Entity and relation vocabularies
- Entity and relation text descriptions
- Downloaded images (cached)
- Metadata

Usage:
    python scrape_mkg_data.py --help
"""
import argparse
import json
import time
import re
import requests
from pathlib import Path
from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image


# ============================================================================
# TRIPLE PARSING
# ============================================================================

def parse_triples(file_path):
    """Parse Wikidata triples."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    triples = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Remove trailing dot
        line = line.rstrip(' .')
        parts = line.split(' ')
        
        if len(parts) != 3:
            print(f"Warning: Skipping invalid triple: {line}")
            continue
            
        h, r, t = parts
        triples.append((h, r, t))
    
    return triples


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_wikidata_uri(uri):
    """
    Extract readable text from Wikidata URI.
    
    Examples:
        http://www.wikidata.org/entity/Q5 → "Q5" (human)
        http://www.wikidata.org/entity/P31 → "P31" (instance of)
    """
    uri_clean = uri.strip('<>')
    
    # Extract Q-number or P-number
    if '/entity/' in uri_clean:
        entity_id = uri_clean.split('/entity/')[-1]
        return entity_id
    elif '/prop/' in uri_clean:
        prop_id = uri_clean.split('/prop/')[-1]
        return prop_id
    
    # Fallback
    return uri_clean.split('/')[-1]


def extract_text_from_wikidata_sparql(uri, timeout=10):
    """Query Wikidata SPARQL endpoint for entity label and description."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        print("⚠️  SPARQLWrapper not installed. Install with: pip install sparqlwrapper")
        return extract_text_from_wikidata_uri(uri)
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setTimeout(timeout)
    
    clean_uri = uri.strip('<>')
    
    # Extract entity ID (Q-number or P-number)
    if '/entity/' in clean_uri:
        entity_id = clean_uri.split('/entity/')[-1]
        entity_uri = f"http://www.wikidata.org/entity/{entity_id}"
    elif '/prop/' in clean_uri:
        entity_id = clean_uri.split('/prop/')[-1]
        entity_uri = f"http://www.wikidata.org/entity/{entity_id}"
    else:
        return extract_text_from_wikidata_uri(uri)
    
    query = f"""
    SELECT ?label ?description WHERE {{
        <{entity_uri}> rdfs:label ?label .
        OPTIONAL {{ <{entity_uri}> schema:description ?description . }}
        FILTER (lang(?label) = 'en')
        FILTER (!bound(?description) || lang(?description) = 'en')
    }}
    LIMIT 1
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        time.sleep(0.1)  # Rate limiting (Wikidata is more permissive)
        results = sparql.query().convert()
        
        if results and "results" in results and "bindings" in results["results"]:
            bindings = results["results"]["bindings"]
            if bindings:
                label = bindings[0].get("label", {}).get("value", "")
                description = bindings[0].get("description", {}).get("value", "")
                
                if label and description:
                    return f"{label}. {description}"
                elif label:
                    return label
                elif description:
                    return description
    except Exception as e:
        pass
    
    # Fallback to URI extraction
    return extract_text_from_wikidata_uri(uri)


# ============================================================================
# IMAGE DOWNLOADING
# ============================================================================

def query_wikidata_for_image(uri, timeout=10):
    """Query Wikidata SPARQL endpoint for entity image URL."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        return None
    
    clean_uri = uri.strip('<>')
    
    # Extract entity ID
    if '/entity/' in clean_uri:
        entity_id = clean_uri.split('/entity/')[-1]
        entity_uri = f"http://www.wikidata.org/entity/{entity_id}"
    else:
        return None
    
    # Only try to get images for entities (Q-numbers), not properties (P-numbers)
    if not entity_id.startswith('Q'):
        return None
    
    query = f"""
    SELECT ?image WHERE {{
        <{entity_uri}> wdt:P18 ?image .
    }}
    LIMIT 1
    """
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setTimeout(timeout)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        time.sleep(0.1)  # Rate limiting
        results = sparql.query().convert()
        
        if results and "results" in results and "bindings" in results["results"]:
            bindings = results["results"]["bindings"]
            if bindings and "image" in bindings[0]:
                return bindings[0]["image"]["value"]
    except Exception as e:
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
    except Exception as e:
        return False


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Scrape MKG data from triple files and Wikidata')
    
    # Input/Output paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw MKG data directory (containing train.txt, valid.txt, test.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--dataset_name', type=str, default='MKG',
                        help='Dataset name (e.g., MKG-W, MKG-Y)')
    
    # Text extraction options
    parser.add_argument('--text_mode', type=str, default='uri_only',
                        choices=['uri_only', 'wikidata_sparql'],
                        help='Text extraction method: uri_only (fast) or wikidata_sparql (slow, detailed)')
    parser.add_argument('--text_sample', type=int, default=None,
                        help='Sample only N entities for Wikidata text queries (for testing)')
    
    # Image downloading options
    parser.add_argument('--download_images', action='store_true',
                        help='Download images from Wikidata (downloads all by default)')
    parser.add_argument('--image_sample', type=int, default=None,
                        help='Limit to first N entities for image downloads (default: None = all entities)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to successfully download (default: None = no limit)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"{args.dataset_name} DATA SCRAPING")
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
            text = extract_text_from_wikidata_uri(entity_uri)
            entity_texts.append(text)
    else:
        print("  Method: Wikidata SPARQL (slow)")
        sample_size = args.text_sample if args.text_sample else len(entities)
        print(f"  Processing {sample_size:,} entities...")
        
        for i, entity_uri in enumerate(tqdm(entities[:sample_size], desc="  Entities")):
            text = extract_text_from_wikidata_sparql(entity_uri)
            entity_texts.append(text)
        
        # Use URI extraction for remaining
        for entity_uri in entities[sample_size:]:
            text = extract_text_from_wikidata_uri(entity_uri)
            entity_texts.append(text)

    
    relation_texts = []
    if args.text_mode == 'wikidata_sparql':
        print("  Extracting relation texts via SPARQL...")
        for relation_uri in tqdm(relations, desc="  Relations"):
            text = extract_text_from_wikidata_sparql(relation_uri)
            relation_texts.append(text)
    else:
        for relation_uri in tqdm(relations, desc="  Relations"):
            text = extract_text_from_wikidata_uri(relation_uri)
            relation_texts.append(text)
    
    print(f"  ✓ Extracted {len(entity_texts):,} entity texts")
    print(f"  ✓ Extracted {len(relation_texts):,} relation texts")
    
    # ========================================================================
    # STEP 4: Download images (optional)
    # ========================================================================
    image_info = {}
    
    if args.download_images:
        print("\n[4/5] Downloading images from Wikidata...")
        
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # Determine which entities to process
        if args.image_sample:
            entities_to_process = entities[:args.image_sample]
            print(f"  Processing sample of {len(entities_to_process):,} entities...")
        else:
            entities_to_process = entities
            print(f"  Processing all {len(entities_to_process):,} entities...")
        
        downloaded_count = 0
        
        for entity_uri in tqdm(entities_to_process, desc="  Querying and downloading"):
            # Stop if we've reached max_images limit
            if args.max_images and downloaded_count >= args.max_images:
                break
            
            entity_id = entity2id[entity_uri]
            image_path = images_dir / f"entity_{entity_id}.jpg"
            
            # Skip if already downloaded
            if image_path.exists():
                image_info[entity_id] = f"images/entity_{entity_id}.jpg"
                downloaded_count += 1
                continue
            
            # Query for image URL
            image_url = query_wikidata_for_image(entity_uri)
            
            if image_url:
                # Try to download
                if download_and_save_image(image_url, image_path):
                    image_info[entity_id] = f"images/entity_{entity_id}.jpg"
                    downloaded_count += 1
        
        print(f"  ✓ Downloaded {len(image_info):,} images")
        print(f"  ✓ Coverage: {len(image_info)/len(entities)*100:.1f}%")
    else:
        print("\n[4/5] Skipping image download")
    
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
        'dataset': args.dataset_name,
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

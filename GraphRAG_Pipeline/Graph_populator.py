import csv
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from typing import Dict, Any, List
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Neo4j credentials
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME") 
PASSWORD = os.getenv("NEO4J_PASSWORD")

class GraphPopulator:
    def __init__(self, batch_size: int = 10000):
        """
        Initialize GraphPopulator with optimized batch size for large datasets
        
        Parameters
        ----------
        batch_size : int, default 10000
            Batch size for processing CSV rows. Optimized for 110K+ datasets:
            - 2000: Original size for 30K datasets
            - 5000: Good for 50-100K datasets  
            - 8000: Optimal for 110K+ datasets with Neo4j
            - 10000: Maximum recommended (may hit memory limits)
        """
        self.graph = Neo4jGraph(
            url=URI,
            username=USER,
            password=PASSWORD,
            refresh_schema=True  # Enable automatic schema refresh
        )
        self.batch_size = batch_size  # Optimized batch size for large datasets
        
    def close(self):
        self.graph._driver.close()
    
    # APOC version checking removed for simplicity
    
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes for optimal performance"""
        # First, drop the problematic text index if it exists
        try:
            self.graph.query("DROP INDEX paragraph_text_fulltext IF EXISTS")
            logger.info("🗑️ Dropped existing problematic text index")
        except Exception as e:
            logger.debug(f"No existing text index to drop: {e}")
        
        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT paragraph_id_unique IF NOT EXISTS FOR (p:Paragraph) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT case_celex_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.celex IS UNIQUE", 
            "CREATE CONSTRAINT year_unique IF NOT EXISTS FOR (y:Year) REQUIRE y.year IS UNIQUE",
            
            # Performance indexes
            "CREATE INDEX paragraph_celex_idx IF NOT EXISTS FOR (p:Paragraph) ON p.celex",
            "CREATE INDEX paragraph_number_idx IF NOT EXISTS FOR (p:Paragraph) ON p.number",
            "CREATE INDEX case_date_idx IF NOT EXISTS FOR (c:Case) ON c.date"
            # Note: Removed text index due to size limitations - GraphRAG will use vector embeddings instead
        ]
        
        for query in constraints_and_indexes:
            try:
                self.graph.query(query)
                logger.info(f"✅ Applied: {query.split()[1]}")
            except Exception as e:
                logger.warning(f"⚠️ Constraint/Index might already exist: {e}")
    
    def create_case_node(self, celex: str, title: str, date: str):
        """Create or update a Case node"""
        query = """
        MERGE (c:Case {celex: $celex})
        SET c.title = $title,
            c.date = date($date),
            c.year = date($date).year
        """
        self.graph.query(query, params={"celex": celex, "title": title, "date": date})
    
    def create_paragraph_node(self, paragraph_id: str, celex: str, number: int, text: str, title: str = ""):
        """Create or update a Paragraph node with clear separate fields"""
        query = """
        MERGE (p:Paragraph {id: $paragraph_id})
        SET p.celex = $celex,
            p.number = $number,
            p.text = $text,
            p.title = $title
        """
        self.graph.query(query, params={
            "paragraph_id": paragraph_id, 
            "celex": celex, 
            "number": number, 
            "text": text,
            "title": title
        })
    
    def create_year_node(self, year: int):
        """Create a Year node for temporal queries"""
        query = """
        MERGE (y:Year {year: $year})
        """
        self.graph.query(query, params={"year": year})
    
    def create_relationships_batch(self, relationships: List[Dict[str, Any]]):
        """Create relationships in batch for better performance"""
        
        # CONTAINS relationships (Case -> Paragraph)
        contains_query = """
        UNWIND $contains_rels as rel
        MATCH (c:Case {celex: rel.case_celex})
        MATCH (p:Paragraph {id: rel.paragraph_id})
        MERGE (c)-[:CONTAINS]->(p)
        """
        
        # CITES relationships (Paragraph -> Paragraph) with self-citation tracking
        cites_query = """
        UNWIND $cites_rels as rel
        MATCH (from:Paragraph {id: rel.from_id})
        MATCH (to:Paragraph {id: rel.to_id})
        MERGE (from)-[:CITES {
            citation_strength: 1.0,
            is_self_citation: rel.is_self_citation,
            created_date: datetime()
        }]->(to)
        """
        
        # DECIDED_IN relationships (Case -> Year)
        decided_in_query = """
        UNWIND $decided_rels as rel
        MATCH (c:Case {celex: rel.celex})
        MATCH (y:Year {year: rel.year})
        MERGE (c)-[:DECIDED_IN]->(y)
        """
        
        # Extract relationship data
        contains_rels = []
        cites_rels = []
        decided_rels = []
        
        for rel in relationships:
            # Detect self-citations by comparing CELEX numbers
            is_self_citation = rel["celex_from"] == rel["celex_to"]
            
            # Case-Paragraph relationships
            contains_rels.extend([
                {"case_celex": rel["celex_from"], "paragraph_id": rel["citing_id"]},
                {"case_celex": rel["celex_to"], "paragraph_id": rel["cited_id"]}
            ])
            
            # Citation relationship with self-citation flag
            cites_rels.append({
                "from_id": rel["citing_id"],
                "to_id": rel["cited_id"],
                "is_self_citation": is_self_citation
            })
            
            # Temporal relationships
            if rel.get("year_from"):
                decided_rels.append({"celex": rel["celex_from"], "year": rel["year_from"]})
            if rel.get("year_to"):
                decided_rels.append({"celex": rel["celex_to"], "year": rel["year_to"]})
        
        # Execute batch operations using LangChain's Neo4jGraph
        if contains_rels:
            self.graph.query(contains_query, params={"contains_rels": contains_rels})
        if cites_rels:
            self.graph.query(cites_query, params={"cites_rels": cites_rels})
        if decided_rels:
            self.graph.query(decided_in_query, params={"decided_rels": decided_rels})
    
    def create_case_to_case_relationships(self):
        """Create case-to-case citation relationships with aggregated statistics"""
        query = """
        // Create CITES_CASE relationships between cases
        MATCH (citing_case:Case)-[:CONTAINS]->(citing_para:Paragraph)-[:CITES]->(cited_para:Paragraph)<-[:CONTAINS]-(cited_case:Case)
        WHERE citing_case <> cited_case  // Avoid self-citations
        
        WITH citing_case, cited_case, 
             count(DISTINCT citing_para) as citing_paragraphs,
             count(DISTINCT cited_para) as cited_paragraphs,
             count(*) as total_citations
        
        MERGE (citing_case)-[r:CITES_CASE]->(cited_case)
        SET r.citation_count = total_citations,
            r.citing_paragraphs = citing_paragraphs,
            r.cited_paragraphs = cited_paragraphs,
            r.authority_weight = toFloat(total_citations) / 10.0,  // Normalized weight
            r.created_date = datetime()
        """
        self.graph.query(query)
        
        # # NOTE: PRECEDES relationship implementation (for later use if required)
        # # This creates temporal ordering relationships between cases showing legal precedence
        # # Earlier cases PRECEDE later cases chronologically
        # """
        # PRECEDES_query = '''
        # // Create PRECEDES relationships for temporal ordering
        # MATCH (earlier_case:Case), (later_case:Case)
        # WHERE earlier_case.date < later_case.date
        # AND exists((earlier_case)-[:CITES_CASE]-(later_case))  // Only connect related cases
        
        # MERGE (earlier_case)-[p:PRECEDES]->(later_case)
        # SET p.temporal_gap_days = duration.between(earlier_case.date, later_case.date).days,
        #     p.temporal_gap_years = duration.between(earlier_case.date, later_case.date).years
        # '''
        # self.graph.query(PRECEDES_query)
        # """
    
    def create_case_statistics(self):
        """Add statistical properties to case nodes including self-citation tracking"""
        query = """
        // Add citation statistics to cases with self-citation tracking
        MATCH (c:Case)
        OPTIONAL MATCH (c)-[:CONTAINS]->(p:Paragraph)
        
        // Count outgoing citations (from this case's paragraphs)
        OPTIONAL MATCH (p)-[out_cites:CITES]->(:Paragraph)
        
        // Count incoming citations (to this case's paragraphs)  
        OPTIONAL MATCH (:Paragraph)-[in_cites:CITES]->(p)
        
        WITH c, 
             count(DISTINCT p) as total_paragraphs,
             count(DISTINCT out_cites) as total_outgoing_citations,
             count(DISTINCT in_cites) as total_incoming_citations,
             count(DISTINCT case when out_cites.is_self_citation = true then out_cites end) as self_citations_out,
             count(DISTINCT case when in_cites.is_self_citation = true then in_cites end) as self_citations_in
        
        SET c.paragraph_count = total_paragraphs,
            c.outgoing_citations = total_outgoing_citations,
            c.incoming_citations = total_incoming_citations,
            c.self_citation_count = self_citations_out + self_citations_in,
            c.external_citation_count = (total_outgoing_citations + total_incoming_citations) - (self_citations_out + self_citations_in),
            c.self_citation_ratio = CASE 
                WHEN (total_outgoing_citations + total_incoming_citations) > 0 
                THEN toFloat(self_citations_out + self_citations_in) / toFloat(total_outgoing_citations + total_incoming_citations)
                ELSE 0.0 
            END,
            c.citation_ratio = CASE 
                WHEN total_paragraphs > 0 THEN toFloat(total_outgoing_citations + total_incoming_citations) / total_paragraphs 
                ELSE 0.0 
            END
        """
        self.graph.query(query)
    
    def parse_date(self, date_str: str) -> tuple:
        """Parse date string and return (date, year)"""
        if not date_str or date_str.strip() == '':
            return None, None
        try:
            # Handle your date format: '1989-02-28'
            date_obj = datetime.strptime(date_str.strip(), '%Y-%m-%d')
            return date_str.strip(), date_obj.year
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")
            return None, None
    
    def process_csv(self, file_path: str):
        """Process the CSV file and populate the graph"""
        logger.info(f"🚀 Starting graph population from {file_path}")
        logger.info(f"⚡ Using optimized batch size: {self.batch_size} (recommended for large datasets)")
        
        # First, create constraints and indexes
        self.create_constraints_and_indexes()
        
        # Count total rows for progress bar
        with open(file_path, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in csv.DictReader(f))
        
        batch_data = []
        cases_created = set()
        years_created = set()
        
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            with tqdm(total=total_rows, desc="Processing rows") as pbar:
                for row in reader:
                    # Generate unique paragraph IDs
                    citing_id = f"{row['CELEX_FROM']}#{row['NUMBER_FROM']}"
                    cited_id = f"{row['CELEX_TO']}#{row['NUMBER_TO']}"
                    
                    # Parse dates
                    date_from, year_from = self.parse_date(row.get('DATE_FROM', ''))
                    date_to, year_to = self.parse_date(row.get('DATE_TO', ''))
                    
                    # Collect batch data
                    batch_item = {
                        'citing_id': citing_id,
                        'cited_id': cited_id,
                        'celex_from': row['CELEX_FROM'],
                        'celex_to': row['CELEX_TO'],
                        'title_from': row['TITLE_FROM'],
                        'title_to': row['TITLE_TO'],
                        'number_from': int(row['NUMBER_FROM']),
                        'number_to': int(row['NUMBER_TO']),
                        'text_from': row['TEXT_FROM'],
                        'text_to': row['TEXT_TO'],
                        'date_from': date_from,
                        'date_to': date_to,
                        'year_from': year_from,
                        'year_to': year_to
                    }
                    
                    batch_data.append(batch_item)
                    
                    # Track unique cases and years
                    cases_created.add((row['CELEX_FROM'], row['TITLE_FROM'], date_from))
                    cases_created.add((row['CELEX_TO'], row['TITLE_TO'], date_to))
                    if year_from:
                        years_created.add(year_from)
                    if year_to:
                        years_created.add(year_to)
                    
                    # Process batch when it reaches the batch size
                    if len(batch_data) >= self.batch_size:
                        self._process_batch(batch_data, cases_created, years_created)
                        batch_data = []
                        cases_created = set()
                        years_created = set()
                    
                    pbar.update(1)
                
                # Process remaining data
                if batch_data:
                    self._process_batch(batch_data, cases_created, years_created)
        
        logger.info("✅ Graph population complete!")
        
        # Create case-to-case relationships and statistics
        logger.info("🔗 Creating case-to-case relationships...")
        self.create_case_to_case_relationships()
        logger.info("📊 Adding case statistics...")
        self.create_case_statistics()
        
        logger.info("✅ Case-level analysis complete!")
        self._print_graph_stats()
    
    def _process_batch(self, batch_data: List[Dict], cases: set, years: set):
        """Process a batch of data using LangChain Neo4jGraph"""
        # Create nodes first
        for celex, title, date in cases:
            if date:  # Only create if we have a valid date
                self.create_case_node(celex, title, date)
        
        for year in years:
            self.create_year_node(year)
        
        for item in batch_data:
            self.create_paragraph_node(
                item['citing_id'], item['celex_from'], 
                item['number_from'], item['text_from'], item['title_from']
            )
            self.create_paragraph_node(
                item['cited_id'], item['celex_to'],
                item['number_to'], item['text_to'], item['title_to']
            )
        
        # Create relationships in batch
        self.create_relationships_batch(batch_data)
    
    def _print_graph_stats(self):
        """Print essential statistics about the created graph"""
        logger.info("📊 Graph Statistics:")
        
        # Basic node and relationship counts - essential for validation
        basic_stats = [
            ("Cases", "MATCH (c:Case) RETURN count(c) as count"),
            ("Paragraphs", "MATCH (p:Paragraph) RETURN count(p) as count"), 
            ("Years", "MATCH (y:Year) RETURN count(y) as count"),
            ("Paragraph Citations", "MATCH ()-[r:CITES]->() RETURN count(r) as count"),
            ("Case Citations", "MATCH ()-[r:CITES_CASE]->() RETURN count(r) as count"),
            ("Contains Relations", "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count")
        ]
        
        for name, query in basic_stats:
            try:
                result = self.graph.query(query)
                count = result[0]['count'] if result else 0
                logger.info(f"   {name}: {count:,}")
            except Exception as e:
                logger.warning(f"   {name}: Error retrieving count - {e}")
        
        # Essential self-citation analysis for graph quality assessment
        try:
            self_citation_analysis = self.graph.query("""
            MATCH ()-[r:CITES]->()
            WITH count(r) as total_citations,
                 count(case when r.is_self_citation = true then 1 end) as self_cites
            RETURN total_citations, self_cites,
                   round(toFloat(self_cites) / total_citations * 100, 2) as self_citation_percentage
            """)
            
            if self_citation_analysis:
                stats = self_citation_analysis[0]
                logger.info("📈 Citation Quality:")
                logger.info(f"   Total Citations: {stats['total_citations']:,}")
                logger.info(f"   Self-Citation Rate: {stats['self_citation_percentage']}%")
                
        except Exception as e:
            logger.warning(f"Citation analysis failed: {e}")

# ==== Usage ====
if __name__ == "__main__":
    # Initialize the graph populator with optimized batch size
    # Choose batch size based on your dataset size:
    # - 2000: For datasets < 50K rows (original default)
    # - 5000: For datasets 50-100K rows 
    # - 8000: For datasets 100K+ rows (recommended for 110K dataset)
    # - 10000: Maximum (may hit memory limits)
    
    populator = GraphPopulator(batch_size=10000)  # Optimized for 110K dataset
    
    try:
        logger.info("🚀 Starting Graph Population...")
        
        # Process your dataset - you can switch between subset and full dataset
        csv_path = "E:/Projects/MSc-Thesis/par-to-par-subset.csv"  # Start with subset for testing
        # csv_path = "E:/Projects/MSc-Thesis/PAR-TO-PAR (MAIN).csv"  # Use this for full dataset (110K rows)
        
        populator.process_csv(csv_path)
        logger.info("✅ Graph population completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        raise
    finally:
        populator.close()
        logger.info("🔌 Connection closed")

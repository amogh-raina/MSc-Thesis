"""
Neo4j Native Vector Store for GraphRAG Pipeline

Uses Neo4j's built-in vector indexing to create embeddings directly from graph content.
Eliminates the need for separate document creation and embedding.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging
import time
from langchain.embeddings.base import Embeddings
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain.schema import Document

logger = logging.getLogger(__name__)


class Neo4jNativeVectorStore:
    """
    Manages vector indices using Neo4j's native vector capabilities.
    
    Instead of creating separate documents and then embedding them,
    this class creates embeddings directly from graph node content.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        embedding: Embeddings,
        index_name: str = "paragraph_vector_index",
        force_rebuild: bool = False,
        embedding_batch_size: int = 2048
    ):
        """
        Initialize Neo4j Native Vector Store
        
        Parameters
        ----------
        neo4j_uri : str
            Neo4j database URI
        neo4j_user : str
            Neo4j username  
        neo4j_password : str
            Neo4j password
        embedding : Embeddings
            LangChain embeddings instance
        index_name : str, default "paragraph_vector_index"
            Name for the vector index
        force_rebuild : bool, default False
            Whether to rebuild existing indices
        embedding_batch_size : int, default 2048
            Batch size for embedding API calls. Optimized for OpenAI text-embedding-3-small:
            - 1000: Original conservative size
            - 1500: Good for most APIs  
            - 2048: Maximum allowed by OpenAI text-embedding-3-small
            - For Tier 1 (3000 RPM): 2048 batch size = ~102K embeddings/second theoretical max
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding = embedding
        self.index_name = index_name
        self.force_rebuild = force_rebuild
        self.embedding_batch_size = embedding_batch_size
        
        # Initialize connections
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )
        
        self.vector_store = None
        
    def setup_vector_index(self) -> None:
        """
        Set up Neo4j vector index using the native vector integration.
        
        This creates a vector store directly from the graph nodes by
        constructing formatted content on-demand from separate fields.
        """
        try:
            logger.info("Setting up Neo4j native vector index...")
            
            # Create Neo4jVector instance that will use existing graph content
            # We construct the formatted content on-demand: (#CELEX:PARA) [TITLE] TEXT
            self.vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embedding,
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                index_name=self.index_name,
                node_label="Paragraph",
                text_node_properties=["text"],  # Use basic text property for setup
                embedding_node_property="embedding"
            )
            
            logger.info(f"✅ Neo4j vector store created with index '{self.index_name}'")
            
        except Exception as e:
            logger.error(f"Failed to setup vector index: {e}")
            raise
    
    def create_embeddings(self) -> None:
        """
        Create embeddings for all paragraph nodes that don't have them.
        
        This processes separate fields and constructs formatted content
        on-demand using the same format as doc_builder.py
        """
        try:
            logger.info("🔍 Checking for paragraphs that need embeddings...")
            
            # First, check embedding status
            status_query = """
            MATCH (p:Paragraph)
            RETURN 
                count(p) as total_paragraphs,
                count(p.embedding) as has_embedding_prop,
                count(case when p.embedding IS NOT NULL then 1 end) as non_null_embeddings
            """
            
            status_result = self.graph.query(status_query)
            if status_result:
                stats = status_result[0]
                total = stats['total_paragraphs']
                has_prop = stats['has_embedding_prop']
                non_null = stats['non_null_embeddings']
                
                logger.info(f"📊 Embedding status: {non_null}/{total} paragraphs have embeddings")
                
                # Check force_rebuild flag
                if self.force_rebuild:
                    logger.info("🔄 Force rebuild enabled - recreating all embeddings")
                elif non_null == total and non_null > 0:
                    logger.info("✅ All paragraphs already have embeddings - skipping creation")
                    return
                elif non_null > 0:
                    logger.info(f"⚠️ {total - non_null} paragraphs missing embeddings")
            
            # Get paragraphs that need embeddings and construct formatted content  
            # For faster testing, you can add LIMIT 5000 to only embed a subset
            if self.force_rebuild:
                # Force rebuild: get ALL paragraphs
                query = """
                MATCH (p:Paragraph)
                WHERE p.text IS NOT NULL
                WITH p,
                     CASE 
                         WHEN p.celex IS NOT NULL AND p.number IS NOT NULL AND p.title IS NOT NULL
                         THEN "(#" + p.celex + ":" + toString(p.number) + ") [" + p.title + "] " + p.text
                         WHEN p.celex IS NOT NULL AND p.number IS NOT NULL  
                         THEN "(#" + p.celex + ":" + toString(p.number) + ") " + p.text
                         ELSE p.text
                     END AS formatted_content
                RETURN p.id as id, formatted_content as content
                // LIMIT 5000  // Uncomment this line for faster testing with subset
                """
            else:
                # Normal mode: only get paragraphs missing embeddings
                query = """
                MATCH (p:Paragraph)
                WHERE p.embedding IS NULL AND p.text IS NOT NULL
                WITH p,
                     CASE 
                         WHEN p.celex IS NOT NULL AND p.number IS NOT NULL AND p.title IS NOT NULL
                         THEN "(#" + p.celex + ":" + toString(p.number) + ") [" + p.title + "] " + p.text
                         WHEN p.celex IS NOT NULL AND p.number IS NOT NULL  
                         THEN "(#" + p.celex + ":" + toString(p.number) + ") " + p.text
                         ELSE p.text
                     END AS formatted_content
                RETURN p.id as id, formatted_content as content
                // LIMIT 5000  // Uncomment this line for faster testing with subset
                """
            
            results = self.graph.query(query)
            
            if not results:
                logger.info("✅ All paragraphs already have embeddings")
                return
            
            total_batches = (len(results) + self.embedding_batch_size - 1) // self.embedding_batch_size  # Ceiling division
            logger.info(f"Creating embeddings for {len(results)} paragraphs in {total_batches} batches...")
            logger.info(f"⚡ Using optimized embedding batch size: {self.embedding_batch_size} (maximum for OpenAI text-embedding-3-small)")
            
            # Calculate theoretical processing speed based on rate limits
            # For OpenAI Tier 1: 3000 RPM = 50 requests/second
            theoretical_speed = 50 * self.embedding_batch_size  # embeddings per second
            logger.info(f"📊 Theoretical speed: {theoretical_speed:,} embeddings/second at 3000 RPM")
            
            # Process in batches
            for i in range(0, len(results), self.embedding_batch_size):
                batch_num = i // self.embedding_batch_size + 1
                batch = results[i:i + self.embedding_batch_size]
                
                logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} paragraphs)...")
                
                # Extract formatted content for embedding
                content_texts = [item['content'] for item in batch]
                paragraph_ids = [item['id'] for item in batch]
                
                # Generate embeddings (this is the slow step)
                logger.info(f"   📡 Calling embedding API for {len(content_texts)} texts...")
                
                # Track embedding API performance
                embed_start_time = time.time()
                embeddings = self.embedding.embed_documents(content_texts)
                embed_duration = time.time() - embed_start_time
                
                logger.info(f"   ✅ Received {len(embeddings)} embeddings in {embed_duration:.2f}s ({len(content_texts)/embed_duration:.1f} texts/sec)")
                
                # Update nodes with embeddings (efficient batch UNWIND update)
                logger.info(f"   💾 Batch updating Neo4j with {len(embeddings)} embeddings...")
                
                # Prepare batch data for UNWIND
                batch_updates = []
                for pid, embedding_vec in zip(paragraph_ids, embeddings):
                    batch_updates.append({
                        "paragraph_id": pid,
                        "embedding": embedding_vec
                    })
                
                # Efficient batch update using UNWIND (much faster than individual queries)
                batch_update_query = """
                UNWIND $batch_updates as update
                MATCH (p:Paragraph {id: update.paragraph_id})
                SET p.embedding = update.embedding
                """
                self.graph.query(batch_update_query, params={"batch_updates": batch_updates})
                
                logger.info(f"✅ Completed batch {batch_num}/{total_batches}")
                
                # Rate limiting optimized for OpenAI Tier 1 (3000 RPM = 50 requests/second)  
                if batch_num < total_batches:  # Don't wait after the last batch
                    # With 3000 RPM, we can make a request every 20ms (1000ms / 50 requests)
                    # Small safety margin: 25ms between requests
                    time.sleep(0.025)  # 25ms pause - much faster than previous 500ms
            
            logger.info(f"✅ Created embeddings for {len(results)} paragraphs")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    def initialize(self) -> None:
        """
        Initialize the complete vector store.
        
        This method:
        1. Creates embeddings for graph content
        2. Sets up Neo4j vector index
        """
        logger.info("🚀 Initializing Neo4j native vector store...")
        
        # Step 1: Create embeddings for paragraph content
        self.create_embeddings()
        
        # Step 2: Setup vector index
        self.setup_vector_index()
        
        logger.info("✅ Neo4j native vector store initialized!")
        self._print_stats()
    
    def get_retriever(self, k: int = 10, search_type: str = "similarity"):
        """
        Get a retriever for vector similarity search
        
        Parameters
        ----------
        k : int, default 10
            Number of results to return
        search_type : str, default "similarity"
            Type of search ('similarity', 'hybrid', etc.)
            
        Returns
        -------
        Retriever instance
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    # REMOVED: similarity_search method - redundant with custom_similarity_search
    # The custom_similarity_search method provides better metadata extraction 
    # and more reliable direct Neo4j queries
    
    # TODO: Implement hybrid search (vector + keyword matching)
    # def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
    #     """
    #     Perform hybrid vector + fulltext search
    #     
    #     Would require:
    #     1. Fulltext index on paragraph text: CREATE FULLTEXT INDEX paragraph_text FOR (p:Paragraph) ON EACH [p.text]
    #     2. Combined scoring mechanism between vector similarity and keyword relevance
    #     3. Proper weighting between vector and keyword scores
    #     """
    #     pass
    
    # Enhanced search moved to separate variant files for better modularity
    
    # Variant 1 search moved to variants/baseline.py for better modularity
    
    def _print_stats(self) -> None:
        """Print vector store statistics"""
        try:
            # Count paragraphs with embeddings
            query = """
            MATCH (p:Paragraph)
            WHERE p.embedding IS NOT NULL
            RETURN count(p) as embedded_count
            """
            result = self.graph.query(query)
            embedded_count = result[0]['embedded_count'] if result else 0
            
            # Total paragraphs
            query = "MATCH (p:Paragraph) RETURN count(p) as total_count"
            result = self.graph.query(query)
            total_count = result[0]['total_count'] if result else 0
            
            logger.info("📊 Vector Store Statistics:")
            logger.info(f"   📄 Total Paragraphs: {total_count:,}")
            logger.info(f"   🧠 Embedded Paragraphs: {embedded_count:,}")
            logger.info(f"   📈 Index Name: {self.index_name}")
            
        except Exception as e:
            logger.warning(f"Could not generate stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics as dictionary"""
        try:
            # Count paragraphs with embeddings
            query = """
            MATCH (p:Paragraph)
            WHERE p.embedding IS NOT NULL
            RETURN count(p) as embedded_count
            """
            result = self.graph.query(query)
            embedded_count = result[0]['embedded_count'] if result else 0
            
            # Total paragraphs
            query = "MATCH (p:Paragraph) RETURN count(p) as total_count"
            result = self.graph.query(query)
            total_count = result[0]['total_count'] if result else 0
            
            return {
                "total_paragraphs": total_count,
                "embedded_paragraphs": embedded_count,
                "index_name": self.index_name,
                "embedding_coverage": embedded_count / total_count if total_count > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Could not generate stats: {e}")
            return {}
    
    def close(self):
        """Close Neo4j connections"""
        if self.graph:
            self.graph._driver.close()

    # Enhanced and variant retrievers moved to separate variant files for better modularity

    def custom_similarity_search(
        self, 
        query: str, 
        k: int = 10,
        with_graph_context: bool = False
    ) -> List[Document]:
        """
        Custom similarity search that directly queries Neo4j vector index.
        
        This bypasses LangChain's limitations with custom retrieval queries
        and ensures proper metadata extraction.
        
        Parameters
        ----------
        query : str
            Search query
        k : int, default 10
            Number of results
        with_graph_context : bool, default False
            Whether to include additional graph context
            
        Returns
        -------
        List[Document]
            Search results with properly extracted metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding.embed_query(query)
            
            # Build the vector similarity search query
            if with_graph_context:
                # Enhanced search that includes citation relationships
                vector_query = f"""
                CALL db.index.vector.queryNodes('{self.index_name}', $k, $query_embedding) 
                YIELD node, score
                
                WITH node, score,
                     CASE 
                         WHEN node.celex IS NOT NULL AND node.number IS NOT NULL AND node.title IS NOT NULL
                         THEN "(#" + node.celex + ":" + toString(node.number) + ") [" + node.title + "] " + node.text
                         WHEN node.celex IS NOT NULL AND node.number IS NOT NULL  
                         THEN "(#" + node.celex + ":" + toString(node.number) + ") " + node.text
                         ELSE node.text
                     END AS formatted_content
                     
                // Join with case for additional context  
                OPTIONAL MATCH (case:Case)-[:CONTAINS]->(node)
                
                RETURN 
                    formatted_content AS text,
                    score,
                    {{
                        celex: node.celex,
                        para_no: toString(node.number),
                        case_title: node.title,
                        paragraph_id: node.id,
                        node_type: "paragraph",
                        
                        // Add citation context
                        cites_count: COUNT {{ (node)-[:CITES]->() }},
                        cited_by_count: COUNT {{ ()-[:CITES]->(node) }},
                        case_year: COALESCE(case.year, date(case.date).year),
                        
                        // Add related paragraphs with proper text field
                        related_paragraphs: [
                            (node)-[:CITES]->(cited) | 
                            {{
                                id: cited.id, 
                                celex: cited.celex,
                                para_no: cited.number,
                                text: cited.text[0..200] + "..."
                            }}
                        ][0..3]
                    }} AS metadata
                ORDER BY score DESC
                """
            else:
                # Basic search with formatted content
                vector_query = f"""
                CALL db.index.vector.queryNodes('{self.index_name}', $k, $query_embedding) 
                YIELD node, score
                
                WITH node, score,
                     CASE 
                         WHEN node.celex IS NOT NULL AND node.number IS NOT NULL AND node.title IS NOT NULL
                         THEN "(#" + node.celex + ":" + toString(node.number) + ") [" + node.title + "] " + node.text
                         WHEN node.celex IS NOT NULL AND node.number IS NOT NULL  
                         THEN "(#" + node.celex + ":" + toString(node.number) + ") " + node.text
                         ELSE node.text
                     END AS formatted_content
                     
                RETURN 
                    formatted_content AS text,
                    score,
                    {{
                        celex: node.celex,
                        para_no: toString(node.number),
                        case_title: node.title,
                        paragraph_id: node.id,
                        node_type: "paragraph",
                        raw_text: node.text,
                        word_count: size(split(node.text, ' ')),
                        char_count: size(node.text)
                    }} AS metadata
                ORDER BY score DESC
                """
            
            # Execute the vector search query
            results = self.graph.query(
                vector_query,
                params={
                    "k": k,
                    "query_embedding": query_embedding
                }
            )
            
            # Convert results to Document objects
            documents = []
            for result in results:
                doc = Document(
                    page_content=result['text'],
                    metadata=result['metadata']
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Custom similarity search failed: {e}")
            # Fallback to LangChain method if custom fails
            logger.warning("Falling back to LangChain similarity search...")
            return self.vector_store.similarity_search(query, k=k)


# Factory functions for easy initialization
def create_neo4j_vector_store(
    embedding: Embeddings,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    index_name: str = "paragraph_vector_index",
    force_rebuild: bool = False,
    auto_initialize: bool = True,
    embedding_batch_size: int = 2048
) -> Neo4jNativeVectorStore:
    """
    Factory function to create Neo4j native vector store
    
    Parameters
    ----------
    embedding : Embeddings
        Embedding model
    neo4j_uri : str, optional
        Neo4j URI (uses environment variable if not provided)
    neo4j_user : str, optional
        Neo4j username (uses environment variable if not provided)
    neo4j_password : str, optional
        Neo4j password (uses environment variable if not provided)
    index_name : str, default "paragraph_vector_index"
        Name for the vector index
    force_rebuild : bool, default False
        Whether to rebuild existing indices
    auto_initialize : bool, default True
        Whether to automatically initialize the vector store
    embedding_batch_size : int, default 2048
        Batch size for embedding API calls (optimized for OpenAI Tier 1)
        
    Returns
    -------
    Neo4jNativeVectorStore
        Configured (and optionally initialized) vector store
    """
    import os
    
    # Get Neo4j credentials from environment if not provided
    neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
    neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
    neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise ValueError("Neo4j credentials must be provided or set in environment variables")
    
    # Create vector store
    vector_store = Neo4jNativeVectorStore(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        embedding=embedding,
        index_name=index_name,
        force_rebuild=force_rebuild,
        embedding_batch_size=embedding_batch_size
    )
    
    # Auto-initialize if requested
    if auto_initialize:
        vector_store.initialize()
    
    return vector_store


# Enhanced and variant factory functions moved to separate variant files for better modularity
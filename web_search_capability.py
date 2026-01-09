#!/usr/bin/env python3
"""
Web Search Capability for GAIA Benchmark
Provides real-time information retrieval for answering GAIA questions
"""

import json
import time
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class WebSearchResult:
    """Web search result"""
    url: str
    title: str
    snippet: str
    relevance_score: float
    timestamp: datetime
    

@dataclass
class SearchQuery:
    """Search query with metadata"""
    query: str
    question: str
    level: int
    query_type: str  # 'factual', 'numerical', 'temporal', 'entity'
    confidence: float
    timestamp: datetime


class WebSearchEngine:
    """
    Web Search Engine for GAIA Benchmark
    Supports multiple search APIs and caching
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}  # URL/Query -> Result cache
        self.cache_size = cache_size
        self.search_count = 0
        self.cache_hits = 0
        
        # Search engine configuration
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'google': self._search_google,
            'bing': self._search_bing,
            'wikipedia': self._search_wikipedia
        }
        
        # Default engine
        self.primary_engine = 'duckduckgo'
    
    def extract_search_query(self, question: str, level: int) -> SearchQuery:
        """
        Extract optimal search query from GAIA question
        
        Args:
            question: The GAIA question
            level: Difficulty level (1, 2, or 3)
            
        Returns:
            SearchQuery with optimized query and metadata
        """
        question_lower = question.lower()
        
        # Determine query type
        query_type = self._classify_query_type(question)
        
        # Extract key entities and terms
        entities = self._extract_entities(question)
        numeric_terms = self._extract_numeric_terms(question)
        temporal_terms = self._extract_temporal_terms(question)
        
        # Build optimized query
        if query_type == 'numerical':
            # For numerical questions, search for the concept + terms
            core_concept = entities[0] if entities else question.split()[:3]
            query = f"{core_concept} {' '.join(numeric_terms)}"
        
        elif query_type == 'temporal':
            # For date/time questions, search for event + time context
            core_concept = entities[0] if entities else question.split()[:3]
            query = f"{core_concept} {' '.join(temporal_terms)}"
        
        elif query_type == 'factual':
            # For factual questions, use key entities
            query = ' '.join(entities[:5]) if entities else question
        
        else:  # entity
            # For entity identification, search for the main entity
            query = ' '.join(entities[:3]) if entities else question
        
        # Calculate confidence based on query quality
        confidence = self._calculate_query_confidence(question, query, entities, query_type)
        
        search_query = SearchQuery(
            query=query,
            question=question,
            level=level,
            query_type=query_type,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        return search_query
    
    def _classify_query_type(self, question: str) -> str:
        """Classify the type of query needed"""
        question_lower = question.lower()
        
        # Numerical query indicators
        numerical_indicators = ['calculate', 'how many', 'count', 'sum', 'total', 'add', 'subtract', 'multiply', 'divide']
        if any(indicator in question_lower for indicator in numerical_indicators):
            return 'numerical'
        
        # Temporal query indicators
        temporal_indicators = ['when', 'what year', 'what date', 'in', 'since', 'ago', 'before', 'after']
        if any(indicator in question_lower for indicator in temporal_indicators):
            return 'temporal'
        
        # Factual query indicators
        factual_indicators = ['what', 'which', 'who', 'where', 'how', 'why', 'name', 'capital', 'located']
        if any(indicator in question_lower for indicator in factual_indicators):
            return 'factual'
        
        return 'entity'
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract key entities from question"""
        # Simple entity extraction using capitalization and quotes
        entities = []
        
        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend(quoted)
        
        # Extract capitalized words (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', question)
        entities.extend(capitalized)
        
        # Remove stopwords
        stopwords = {'What', 'Which', 'How', 'When', 'Where', 'The', 'A', 'An', 'In', 'On', 'At', 'Of'}
        entities = [e for e in entities if e not in stopwords]
        
        return entities
    
    def _extract_numeric_terms(self, question: str) -> List[str]:
        """Extract numeric terms from question"""
        # Find numbers and units
        numeric = re.findall(r'\d+\.?\d*', question)
        return numeric
    
    def _extract_temporal_terms(self, question: str) -> List[str]:
        """Extract temporal terms from question"""
        question_lower = question.lower()
        
        temporal_terms = []
        for word in question.split():
            if word in ['january', 'february', 'march', 'april', 'may', 'june', 
                       'july', 'august', 'september', 'october', 'november', 'december',
                       'year', 'month', 'day', 'date', 'time', 'century', 'decade']:
                temporal_terms.append(word)
        
        return temporal_terms
    
    def _calculate_query_confidence(self, question: str, query: str, entities: List[str], query_type: str) -> float:
        """Calculate confidence in query quality"""
        confidence = 0.5  # Base confidence
        
        # Boost for entity presence
        if entities:
            confidence += 0.2
        
        # Boost for specific query types
        if query_type in ['numerical', 'factual', 'temporal']:
            confidence += 0.1
        
        # Boost for query length
        if 3 <= len(query.split()) <= 6:
            confidence += 0.1
        
        # Boost for containing key question terms
        question_words = set(question.lower().split())
        query_words = set(query.lower().split())
        overlap = len(question_words & query_words) / len(question_words)
        confidence += overlap * 0.1
        
        return min(1.0, confidence)
    
    def search(self, query: SearchQuery, num_results: int = 5) -> List[WebSearchResult]:
        """
        Perform web search with caching
        
        Args:
            query: SearchQuery object
            num_results: Number of results to return
            
        Returns:
            List of WebSearchResult objects
        """
        self.search_count += 1
        
        print(f"🔍 Searching: {query.query} (Query type: {query.query_type}, Confidence: {query.confidence:.2f})")
        
        # Check cache
        cache_key = self._get_cache_key(query.query)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Perform search using primary engine
        search_func = self.search_engines[self.primary_engine]
        results = search_func(query.query, num_results)
        
        # Cache results
        self.cache[cache_key] = results
        
        # Limit cache size
        if len(self.cache) > self.cache_size:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:len(self.cache) - self.cache_size]
            for key in oldest_keys:
                del self.cache[key]
        
        return results
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _search_duckduckgo(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Search using DuckDuckGo (no API key required)"""
        try:
            # DuckDuckGo API (free, no key required)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for i, result in enumerate(data.get('RelatedTopics', [])[:num_results]):
                search_result = WebSearchResult(
                    url=result.get('FirstURL', ''),
                    title=result.get('Text', ''),
                    snippet=result.get('Text', '')[:200],
                    relevance_score=1.0 - (i * 0.1),
                    timestamp=datetime.now()
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            print(f"⚠️ DuckDuckGo search error: {e}")
            return []
    
    def _search_google(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Search using Google (requires API key)"""
        # Placeholder for Google Custom Search API
        # You would need to get an API key from Google Cloud Console
        print("⚠️ Google search requires API key. Using fallback...")
        return self._search_duckduckgo(query, num_results)
    
    def _search_bing(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Search using Bing (requires API key)"""
        # Placeholder for Bing Search API
        print("⚠️ Bing search requires API key. Using fallback...")
        return self._search_duckduckgo(query, num_results)
    
    def _search_wikipedia(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Search using Wikipedia API (free)"""
        try:
            # Wikipedia API
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'utf8': '',
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            search_data = data.get('query', {})
            for i, result in enumerate(search_data.get('search', [])[:num_results]):
                search_result = WebSearchResult(
                    url=f"https://en.wikipedia.org/wiki/{result.get('title', '').replace(' ', '_')}",
                    title=result.get('title', ''),
                    snippet=result.get('snippet', '')[:200],
                    relevance_score=1.0 - (i * 0.1),
                    timestamp=datetime.now()
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            print(f"⚠️ Wikipedia search error: {e}")
            return []
    
    def extract_answer_from_results(self, results: List[WebSearchResult], search_query: SearchQuery) -> Optional[str]:
        """
        Extract answer from search results based on query type
        
        Args:
            results: List of search results
            search_query: Original SearchQuery object
            
        Returns:
            Extracted answer or None
        """
        if not results:
            return None
        
        question_lower = search_query.question.lower()
        query_type = search_query.query_type
        
        # For numerical queries, extract numbers
            for result in results:
                # Try to find answer in snippet or title
                text_to_search = result.snippet + ' ' + result.title
                numbers = re.findall(r'\d+\.?\d*', text_to_search)
                if numbers:
                    # Find most reasonable answer
                    # Prefer numbers that match the question pattern
                    question_lower = search_query.question.lower()
                    
                    # Check for calculation indicators
                    if 'multiply' in question_lower or '*' in question_lower:
                        # For multiplication, look for 2-3 numbers
                        nums = [float(n) for n in numbers]
                        # Try to find 2-3 numbers
                        if len(nums) >= 2 and question_lower.count('*') == 1:
                            # Multiplication
                            result_value = nums[0] * nums[1]
                            return str(int(result_value)) if result_value.is_integer() else f"{result_value:.2f}"
                    
                    # Return largest number if no clear calculation
                    numbers_sorted = sorted([float(n) for n in numbers])
                    return str(int(numbers_sorted[-1])) if numbers_sorted[-1].is_integer() else numbers_sorted[-1]
        
        # For temporal queries, extract dates/years
        elif query.query_type == 'temporal':
            for result in results:
                years = re.findall(r'\b(19|20)\d{2}\b', result.snippet + result.title)
                if years:
                    return years[-1]
                dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', result.snippet + result.title)
                if dates:
                    return dates[-1]
        
        # For factual queries, extract key information
        elif query.query_type == 'factual':
            # Look for patterns like "X is Y", "X was Y", "capital is Y"
            patterns = [
                r'(\w+)\s+(?:is|was|became)\s+(\w+)',
                r'capital\s+(?:is|of)\s+(\w+)',
                r'(?i)answer\s*[:=]\s*(\w+)'
            ]
            
            for result in results:
                text = result.snippet + ' ' + result.title
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        return matches[-1][-1]  # Return the answer part
        
        # For entity queries, return the most relevant entity
        else:
            return results[0].title  # Return title of top result
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        cache_hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
        return {
            'search_count': self.search_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'primary_engine': self.primary_engine,
            'cache_hit_rate_pct': f"{cache_hit_rate:.1%}%"
        }


class GAIAQuestionAnswering:
    """
    GAIA Question Answering with Web Search
    Combines local reasoning with web search for accurate answers
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.search_engine = WebSearchEngine()
        self.answer_history = []
        
        # Answer confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def answer_question(self, question: str, level: int, use_search: bool = True) -> Dict[str, Any]:
        """
        Answer a GAIA question with optional web search
        
        Args:
            question: The GAIA question
            level: Difficulty level (1, 2, or 3)
            use_search: Whether to use web search
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Extract search query
        search_query = self.search_engine.extract_search_query(question, level)
        
        # Perform web search if enabled
        search_results = []
        web_answer = None
        
        if use_search and level >= 2:  # Use search for level 2 and 3 questions
            print(f"🔍 Searching web for: {search_query.query}")
            search_results = self.search_engine.search(search_query, num_results=5)
            
            # Extract answer from search results
            web_answer = self.search_engine.extract_answer_from_results(search_results, search_query)
            
            if web_answer:
                print(f"✓ Found potential answer: {web_answer}")
        
        # Combine with local reasoning
        final_answer = self._combine_reasoning_and_search(question, web_answer, level)
        
        # Calculate confidence
        confidence = self._calculate_confidence(question, final_answer, search_results, level)
        
        execution_time = time.time() - start_time
        
        # Build result
        result = {
            'question': question,
            'level': level,
            'search_query': search_query.query,
            'query_type': search_query.query_type,
            'search_results_used': len(search_results),
            'web_answer': web_answer,
            'final_answer': final_answer,
            'confidence': confidence,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.answer_history.append(result)
        
        return result
    
    def _combine_reasoning_and_search(self, question: str, web_answer: Optional[str], level: int) -> str:
        """
        Combine local reasoning with web search results
        """
        if web_answer:
            # For level 2 and 3, trust web search more
            if level >= 2:
                return str(web_answer)
            else:
                # For level 1, use web answer as fallback
                return web_answer
        
        # Fallback to local reasoning
        return self._local_reasoning(question, level)
    
    def _local_reasoning(self, question: str, level: int) -> str:
        """Local reasoning fallback"""
        # Simple heuristic reasoning
        question_lower = question.lower()
        
        # Numerical reasoning
        numbers = re.findall(r'\d+\.?\d*', question)
        if numbers and len(numbers) >= 2:
            if 'add' in question_lower or 'sum' in question_lower:
                return str(sum(float(n) for n in numbers))
            elif 'multiply' in question_lower:
                result = 1
                for n in numbers:
                    result *= float(n)
                return str(int(result) if result.is_integer() else result)
        
        # Fallback
        return "unknown"
    
    def _calculate_confidence(self, question: str, answer: str, search_results: List[WebSearchResult], level: int) -> float:
        """Calculate confidence in answer"""
        confidence = 0.3  # Base confidence
        
        # Boost for search results
        if search_results:
            confidence += 0.4
        
        # Boost for level 1 questions (simpler)
        if level == 1:
            confidence += 0.1
        
        # Boost for specific answer format
        if answer and answer != "unknown":
            confidence += 0.2
        
        return min(1.0, confidence)


def test_web_search():
    """Test web search capability"""
    print("="*80)
    print("TESTING WEB SEARCH CAPABILITY")
    print("="*80)
    
    # Initialize search engine
    search_engine = WebSearchEngine()
    
    # Initialize QA system
    qa_system = GAIAQuestionAnswering()
    
    # Test questions
    test_questions = [
        ("What is the capital of France?", 1),
        ("How many planets are in the solar system?", 1),
        ("When was the first iPhone released?", 2),
        ("Calculate 15 * 3", 1)
    ]
    
    print(f"\n🧪 Testing {len(test_questions)} questions...\n")
    
    for i, (question, level) in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        
        # Answer with search
        result = qa_system.answer_question(question, level, use_search=True)
        
        print(f"Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print(f"Search results: {result['search_results_used']}")
        print()
    
    # Print statistics
    stats = search_engine.get_statistics()
    print("="*80)
    print("SEARCH ENGINE STATISTICS")
    print("="*80)
    print(f"Total searches: {stats['search_count']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Cache size: {stats['cache_size']}")
    print(f"Primary engine: {stats['primary_engine']}")
    print("="*80)


if __name__ == "__main__":
    test_web_search()

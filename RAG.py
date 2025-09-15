import sqlite3
from typing import List, Dict
from typing import Tuple


class ExperimentRAG:
    def __init__(self, db_path: str = 'funsearch.db'):
        """Initialize the RAG system with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def get_top_policies_by_cache_hit(self, workload: str, top_n: int = 2) -> List[Dict]:
        """
        Retrieve top N policies for a given workload based on cache hit rate
        
        Args:
            workload: The workload to query
            top_n: Number of top policies to return (default: 2)
            
        Returns:
            List of dictionaries containing policy information with:
            - policy name
            - policy description
            - workload description
            - cpp file path
            - cache hit rate
        """
        query = '''
        SELECT 
            policy, 
            policy_description, 
            workload_description, 
            cpp_file_path,
            cache_hit_rate
        FROM experiments
        WHERE workload = ?
        ORDER BY cache_hit_rate DESC
        LIMIT ?
        '''
        
        self.cursor.execute(query, (workload, top_n))
        results = self.cursor.fetchall()
        
        # Convert results to list of dictionaries
        policies = []
        for row in results:
            policies.append({
                'policy': row[0],
                'policy_description': row[1],
                'workload_description': row[2],
                'cpp_file_path': row[3],
                'cache_hit_rate': row[4]
            })
        
        return policies
    
    def get_top_policies_by_score(self, workload: str, top_n: int = 2) -> List[Dict]:
        """
        Retrieve top N policies for a given workload based on score
        
        Args:
            workload: The workload to query
            top_n: Number of top policies to return (default: 2)
            
        Returns:
            List of dictionaries containing policy information with:
            - policy name
            - policy description
            - workload description
            - cpp file path
            - score
        """
        query = '''
        SELECT 
            policy, 
            policy_description, 
            workload_description, 
            cpp_file_path,
            score
        FROM experiments
        WHERE workload = ?
        ORDER BY score DESC
        LIMIT ?
        '''
        
        self.cursor.execute(query, (workload, top_n))
        results = self.cursor.fetchall()
        
        # Convert results to list of dictionaries
        policies = []
        for row in results:
            policies.append({
                'policy': row[0],
                'policy_description': row[1],
                'workload_description': row[2],
                'cpp_file_path': row[3],
                'score': row[4]
            })
        
        return policies
    
    def get_all_workloads_with_description(self) -> str:
        """
        Retrieve all distinct workloads and their descriptions as a formatted string.
        
        Returns:
            A string in the format:
            "workload1: description1\nworkload2: description2\n..."
        """
        query = '''
        SELECT DISTINCT workload, workload_description
        FROM experiments
        ORDER BY workload
        '''
        
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        
        # Format into "workload: description" lines
        result = '\n'.join(f"{workload}: {description}" for workload, description in rows)
    
        return result

    def get_all_workloads_with_description_and_traces(self) -> Tuple[str, List[Dict[str, str]]]:
        """
        Retrieve:
        1. A formatted string of all distinct workloads and their descriptions
        2. A list of trace dictionaries with 'name' and 'trace_path'

        Returns:
            A tuple of:
                - Formatted string: "workload1: description1\nworkload2: description2\n..."
                - List of dicts: [{"name": workload1, "trace_path": cpp_file_path}, ...]
        """
        query = '''
        SELECT DISTINCT workload, workload_description, cpp_file_path
        FROM experiments
        ORDER BY workload
        '''
        
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        # For workload descriptions
        workload_lines = []
        trace_list = []
        seen = set()  # Ensure uniqueness by workload (for string part) and (workload, cpp_file_path) for traces

        for workload, description, cpp_file_path in rows:
            if workload not in seen:
                workload_lines.append(f"{workload}: {description}")
                seen.add(workload)
            trace_list.append({"name": workload, "trace_path": cpp_file_path})

        return '\n'.join(workload_lines), trace_list

    def generate_response(self, workload: str) -> str:
        """
        Generate a natural language response with the top policies for a workload
        
        Args:
            workload: The workload to query
            
        Returns:
            Formatted response string
        """
        top_policies = self.get_top_policies_by_cache_hit(workload)
        
        if not top_policies:
            return f"No data available for workload: {workload}"
        
        response = f"Workload: {workload}\n"
        response += f"Description: {top_policies[0]['workload_description']}\n\n"
        response += f"Top {len(top_policies)} policies by cache hit rate:\n\n"
        
        for i, policy in enumerate(top_policies, 1):
            response += f"{i}. Policy: {policy['policy']}\n"
            response += f"   Description: {policy['policy_description']}\n"
            response += f"   Cache Hit Rate: {policy['cache_hit_rate']:.2%}\n"
            response += f"   CPP File Path: {policy['cpp_file_path']}\n\n"
        
        return response
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Example usage
if __name__ == "__main__":
    rag = ExperimentRAG()
    
    # Example query for a specific workload
    workload_query = "example_workload"  # Replace with your actual workload
    response = rag.generate_response(workload_query)
    print(response)
    
    rag.close()
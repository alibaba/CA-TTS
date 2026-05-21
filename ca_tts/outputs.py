"""
Output classes for CA-TTS
(Simplified from the original DeepThinkLLM output class)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class DeepThinkOutput:
    """Output container for refactored deep thinking results"""

    # Primary results
    final_answer: Optional[str] = None

    # Multiple voting results (from self-consistency)
    voting_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Traces (contains all generated traces)
    all_traces: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_traces_count: int = 0

    # Token statistics
    total_tokens: int = 0
    avg_tokens_per_trace: float = 0.0

    # Timing information
    generation_time: float = 0.0    # Time for generating samples (e.g., in self-consistency)
    processing_time: float = 0.0    # Time for voting or correction steps
    total_time: float = 0.0

    # Configuration used (can be populated by strategy functions)
    config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    mode: str = "strategy" # 'self_consistency', 'self_correction', etc.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        if self.total_traces_count > 0:
            self.avg_tokens_per_trace = self.total_tokens / self.total_traces_count
        else:
            self.avg_tokens_per_trace = 0.0
            
        return {
            "final_answer": self.final_answer,
            "voting_results": self.voting_results,
            "all_traces": self.all_traces,
            "total_traces_count": self.total_traces_count,
            
            "token_stats": {
                "total_tokens": self.total_tokens,
                "avg_tokens_per_trace": self.avg_tokens_per_trace,
            },
            
            "timing_stats": {
                "generation_time": self.generation_time,
                "processing_time": self.processing_time,
                "total_time": self.total_time,
            },
            
            "config": self.config,
            "mode": self.mode,
            "timestamp": self.timestamp,
        }
    
    def print_summary(self):
        """Print a formatted summary of the results"""
        print(f"\n=== Deep Thinking Summary ===")
        print(f"Strategy Mode: {self.mode}")
        
        if self.mode == "self_correction":
            print(f"Total steps (traces): {self.total_traces_count}")
        else:
            print(f"Total traces generated: {self.total_traces_count}")
        
        if self.final_answer:
            print(f" {self.final_answer}")
        
        print(f"Total tokens generated: {self.total_tokens}")
        
        if self.generation_time > 0:
            print(f"Generation time: {self.generation_time:.2f}s")
            if self.generation_time > 0.01:
                throughput = self.total_tokens / self.generation_time
                print(f"Generation throughput: {throughput:.1f} tokens/second")
        
        if self.processing_time > 0:
            print(f"Processing (voting/correction) time: {self.processing_time:.2f}s")

        print(f"Total time: {self.total_time:.2f}s")
        
        # Print voting results summary (for self-consistency)
        if self.voting_results:
            print(f"\n=== Voting Results Summary ===")
            for method, result in self.voting_results.items():
                if result and result.get('answer'):
                    num_votes = result.get('num_votes', 0)
                    answer_preview = str(result['answer'])[:50] + "..." if len(str(result['answer'])) > 50 else str(result['answer'])
                    print(f"  {method:<25}: {answer_preview} [{num_votes} votes]")
    
    def print_detailed_voting_results(self):
        """Print detailed voting results in table format."""
        if not self.voting_results:
            print("No voting results available.")
            return

        print("-" * 60)
        print(f"{'Method':<30} {'Votes':<6} {'Answer'}")
        print("-" * 60)

        for method, result in self.voting_results.items():
            if result and result.get('answer') is not None:
                answer = str(result.get('answer', 'None'))
                # Truncate long answers for table display
                answer_preview = (answer[:40] + '...') if len(answer) > 43 else answer
                num_votes = result.get('num_votes', 0)

                print(f"{method:<30} {num_votes:<6} {answer_preview}")
            else:
                # Handle cases where voting method may fail and return None
                print(f"{method:<30} {'-':<6} {'No answer generated'}")

        print("-" * 60)
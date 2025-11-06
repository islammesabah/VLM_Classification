from datetime import datetime

class OpenRouterMonitor:
    def __init__(self):
        # Usage tracking
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.session_start = datetime.now()
        
        # Per-call tracking
        self.call_history = []
    
    def track_call(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """Track a single API call"""
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        
        # Store call details
        call_info = {
            'request_num': self.total_requests,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'timestamp': datetime.now()
        }
        self.call_history.append(call_info)
        
        # Display current call info
        self._display_call_info(call_info)
    
    def _display_call_info(self, call_info: dict):
        """Display information about the current call"""
        print("\n" + "="*60)
        print(f"📊 API CALL #{call_info['request_num']} USAGE REPORT")
        print("="*60)
        print(f"\n🔹 This Call:")
        print(f"   Prompt tokens: {call_info['prompt_tokens']:,}")
        print(f"   Completion tokens: {call_info['completion_tokens']:,}")
        print(f"   Total tokens: {call_info['total_tokens']:,}")
        
        print(f"\n🔹 Session Totals:")
        print(f"   Total requests: {self.total_requests}")
        print(f"   Total prompt tokens: {self.total_prompt_tokens:,}")
        print(f"   Total completion tokens: {self.total_completion_tokens:,}")
        print(f"   Total tokens: {self.total_tokens:,}")
        
        duration = (datetime.now() - self.session_start).total_seconds()
        print(f"   Session duration: {duration:.1f}s")
        print(f"   Avg tokens/request: {self.total_tokens/self.total_requests:.0f}")
        print("="*60 + "\n")
    
    def get_session_summary(self) -> dict:
        """Get a comprehensive summary of the session"""
        duration = (datetime.now() - self.session_start).total_seconds()
        return {
            'total_requests': self.total_requests,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'session_duration_seconds': duration,
            'avg_tokens_per_request': self.total_tokens / self.total_requests if self.total_requests > 0 else 0,
            'requests_per_minute': (self.total_requests / duration) * 60 if duration > 0 else 0
        }
    
    def print_summary(self):
        """Print a formatted summary of the session"""
        summary = self.get_session_summary()
        print("\n" + "="*60)
        print("📈 FINAL SESSION SUMMARY")
        print("="*60)
        print(f"Total API Calls: {summary['total_requests']}")
        print(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}")
        print(f"Total Completion Tokens: {summary['total_completion_tokens']:,}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Session Duration: {summary['session_duration_seconds']:.1f}s")
        print(f"Average Tokens/Request: {summary['avg_tokens_per_request']:.0f}")
        print(f"Requests per Minute: {summary['requests_per_minute']:.2f}")
        print("="*60 + "\n")

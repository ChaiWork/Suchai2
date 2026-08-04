import sys
import time


class ProgressBar:
    """Helper class to display training/evaluation progress in terminal with time estimation."""
    def __init__(self, count: int, text: str):
        self.count = count
        self.text = text.ljust(30)
        self.start_time = time.time()

    def update(self, current: int, suffix: str = ""):
        percent = min(100, 100 * current // self.count) if self.count > 0 else 100
        elapsed = time.time() - self.start_time
        
        # Estimate remaining time
        if current > 0:
            avg_time_per_item = elapsed / current
            est_total_time = avg_time_per_item * self.count
            est_remaining = est_total_time - elapsed
            
            elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
            rem_min, rem_sec = divmod(int(max(0, est_remaining)), 60)
            time_str = f"[{elapsed_min:02d}:{elapsed_sec:02d}<{rem_min:02d}:{rem_sec:02d}, {avg_time_per_item:.1f}s/game]"
        else:
            time_str = f"[00:00<--:--, --s/game]"
            
        # 15-char width progress bar
        bar_width = 15
        filled_width = int(bar_width * current // self.count) if self.count > 0 else bar_width
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        suffix_str = f" | {suffix}" if suffix else ""
        sys.stderr.write(f"\r{self.text} {bar} {current}/{self.count} ({percent}%) {time_str}{suffix_str}   ")
        sys.stderr.flush()
        if current >= self.count:
            sys.stderr.write("\n")
            sys.stderr.flush()

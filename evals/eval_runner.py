"""
evals/eval_runner.py

Test runner for the Moms Verdict pipeline.
Validates the output of the pipeline against the assertions
defined in test_cases.json.

Usage (in Colab):
    python evals/eval_runner.py
"""

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import sys
import os

# Add src to path so we can import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import MomsVerdictPipeline

console = Console()

def load_test_cases():
    with open("evals/test_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_assertion(verdict, case):
    """
    Evaluates a specific assertion on a MomsVerdict object.
    Returns (bool passed, str actual_value_for_logging)
    """
    atype = case["assertion_type"]
    expected = case["expected_value"]
    
    if atype == "confidence_level":
        actual = verdict.confidence_level.value
        return actual == expected, actual
        
    elif atype == "confidence_level_in":
        actual = verdict.confidence_level.value
        return actual in expected, actual
        
    elif atype == "fake_review_flagged":
        actual = verdict.fake_review_flag.flagged
        return actual == expected, str(actual)
        
    elif atype == "similarity_score_min":
        actual = verdict.fake_review_flag.average_similarity_score
        return actual >= expected, f"{actual:.2f}"
        
    elif atype == "confidence_score_max":
        actual = verdict.confidence_score
        return actual <= expected, f"{actual:.2f}"
        
    elif atype == "verdicts_empty":
        is_empty = (verdict.verdict_en == "" and verdict.verdict_ar == "")
        actual = "Empty" if is_empty else "Not Empty"
        return is_empty == expected, actual
        
    elif atype == "overall_sentiment":
        actual = verdict.overall_sentiment.value
        return actual == expected, actual
        
    elif atype == "arabic_validation":
        # Check if verdict_ar has arabic characters
        arabic_chars = sum(1 for c in verdict.verdict_ar if '\u0600' <= c <= '\u06FF')
        actual = arabic_chars > 10
        return actual == expected, f"{arabic_chars} arabic chars"
        
    elif atype == "quotes_exist":
        # Check all pros and cons
        all_items = verdict.pros + verdict.cons
        if not all_items:
            return True, "No items"
        missing_quotes = [item for item in all_items if not item.evidence.representative_quote]
        actual = len(missing_quotes) == 0
        return actual == expected, "Quotes missing" if not actual else "All grounded"
        
    elif atype == "max_pros_cons":
        valid = len(verdict.pros) <= expected and len(verdict.cons) <= expected
        actual = f"{len(verdict.pros)} pros, {len(verdict.cons)} cons"
        return valid, actual

    return False, "Unknown assertion type"

def main():
    console.print("\n[bold blue]🚀 Starting Moms Verdict Evals[/bold blue]\n")
    
    test_cases = load_test_cases()
    pipeline = MomsVerdictPipeline()
    
    # Pre-run pipeline on all available JSONs to cache the verdicts
    sample_dir = Path("data/sample_products")
    if not sample_dir.exists() or not list(sample_dir.glob("*.json")):
        console.print("[red]❌ No sample products found. Please run `python data/generate_reviews.py` first.[/red]")
        sys.exit(1)
        
    verdicts = {}
    for json_file in sample_dir.glob("*.json"):
        console.print(f"Running pipeline on {json_file.name}...")
        try:
            verdict = pipeline.run_from_file(str(json_file))
            # Match on product_name or specific file traits if needed. 
            # We'll key by product_name to match test_cases.
            verdicts[verdict.product_name] = verdict
        except Exception as e:
            console.print(f"[red]Pipeline failed on {json_file.name}: {e}[/red]")
            
    # Run test cases
    table = Table(title="Evaluation Results")
    table.add_column("ID", style="cyan")
    table.add_column("Product", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Actual", style="dim")
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        target = case["target_product"]
        
        # Determine which verdicts to run this case against
        targets_to_run = []
        if target == "ALL":
            targets_to_run = list(verdicts.values())
        elif target in verdicts:
            targets_to_run = [verdicts[target]]
        else:
            # Try partial matching if exact name doesn't match perfectly
            for name, v in verdicts.items():
                if target.lower() in name.lower():
                    targets_to_run.append(v)
                    
        if not targets_to_run:
            table.add_row(case["id"], target, case["description"], "[yellow]SKIPPED (No Data)[/yellow]", "-")
            total -= 1
            continue
            
        all_passed = True
        actuals = []
        for v in targets_to_run:
            p, a = evaluate_assertion(v, case)
            if not p:
                all_passed = False
            actuals.append(a)
            
        final_actual = ", ".join(set(actuals))
        if all_passed:
            passed += 1
            table.add_row(case["id"], target, case["description"], "[green]PASS[/green]", final_actual)
        else:
            table.add_row(case["id"], target, case["description"], "[red]FAIL[/red]", final_actual)
            
    console.print(table)
    
    console.print(f"\n[bold]Summary: {passed}/{total} tests passed.[/bold]")
    if passed == total:
        console.print("[bold green]🎉 All evaluations passed![/bold green]")
    else:
        console.print("[bold red]❌ Some evaluations failed. Check the logs.[/bold red]")

if __name__ == "__main__":
    main()

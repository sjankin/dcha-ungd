#!/usr/bin/env python3
"""
Run LLM baseline for DCHA-UNGD Extended benchmark (1989-2024).

Supports OpenAI, DeepSeek, Anthropic, and Google models with structured output.
Uses the prompt templates from prompts/v1/.

Outputs:
- predictions.jsonl / predictions.csv
- raw_responses.jsonl (full API responses)
- failures.jsonl (parse/validation failures)
- cost_latency.csv (per-call usage & timing)
- run_log.jsonl (per-call log)
- manifest.yaml
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# Paths - Modified for extended dataset
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / 'data' / 'benchmark' / 'dcha_ungd_extended_v1'
PROMPTS_DIR = ROOT_DIR / 'prompts' / 'v1'
RUNS_DIR = ROOT_DIR / 'runs'

# Load environment variables
load_dotenv(ROOT_DIR / '.env')


def load_prompts(include_fewshot: bool = False) -> dict:
    """Load prompt templates and optionally few-shot examples."""
    prompts = {
        'system': (PROMPTS_DIR / 'system.txt').read_text().strip(),
        'user_template': (PROMPTS_DIR / 'user_template.txt').read_text().strip(),
        'repair': (PROMPTS_DIR / 'repair_prompt.txt').read_text().strip(),
        'schema': json.loads((PROMPTS_DIR / 'schema.json').read_text()),
        'fewshot_examples': []
    }

    if include_fewshot:
        fewshot_file = PROMPTS_DIR / 'fewshot_train.jsonl'
        if fewshot_file.exists():
            with open(fewshot_file) as f:
                for line in f:
                    prompts['fewshot_examples'].append(json.loads(line))
            print(f"  Loaded {len(prompts['fewshot_examples'])} few-shot examples")

    return prompts


def load_candidates(split: str) -> pd.DataFrame:
    """Load candidate sentences for a given split."""
    gold_df = pd.read_csv(DATA_DIR / 'candidates_gold.csv')

    with open(DATA_DIR / 'splits.json') as f:
        splits = json.load(f)

    split_ids = set(splits['splits'][split]['candidate_ids'])
    return gold_df[gold_df['candidate_id'].isin(split_ids)]


def validate_output(output: dict, sentence: str) -> tuple[bool, str]:
    """Validate LLM output against schema and sentence."""
    required = ['attrib', 'cause_span', 'effect_span', 'link_type', 'rationale_short']
    for field in required:
        if field not in output:
            return False, f"Missing field: {field}"

    # Validate link_type enum
    valid_types = ['NO_CAUSAL_EXTRACTION', 'OTHER_UNCLEAR', 'C2H_HARM', 'C2H_COBEN', 'H2C_JUST']
    if output['link_type'] not in valid_types:
        return False, f"Invalid link_type: {output['link_type']}"

    # Validate spans are substrings
    if output['attrib']:
        if output['cause_span'] and output['cause_span'] not in sentence:
            return False, f"cause_span not in sentence: {output['cause_span'][:50]}..."
        if output['effect_span'] and output['effect_span'] not in sentence:
            return False, f"effect_span not in sentence: {output['effect_span'][:50]}..."

    # Validate consistency
    if not output['attrib']:
        if output['cause_span'] is not None or output['effect_span'] is not None:
            return False, "attrib=false but spans are not null"
        if output['link_type'] != 'NO_CAUSAL_EXTRACTION':
            return False, f"attrib=false but link_type={output['link_type']}"

    return True, ""


def call_api_with_timing(client, model: str, messages: list, schema: dict, provider: str, system_prompt: str = None) -> dict:
    """Call API and return response with timing and usage info."""
    start_time = time.time()
    http_status = 200
    error_message = None

    try:
        if provider == 'openai':
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                },
                temperature=0
            )
            latency_ms = (time.time() - start_time) * 1000
            usage = response.usage
            tokens_input = usage.prompt_tokens if usage else 0
            tokens_output = usage.completion_tokens if usage else 0
            content = response.choices[0].message.content
            raw_response = {
                'id': response.id,
                'model': response.model,
                'content': content,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': tokens_input,
                    'completion_tokens': tokens_output,
                    'total_tokens': tokens_input + tokens_output
                }
            }

        elif provider == 'deepseek':
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_object"
                },
                temperature=0
            )
            latency_ms = (time.time() - start_time) * 1000
            usage = response.usage
            tokens_input = usage.prompt_tokens if usage else 0
            tokens_output = usage.completion_tokens if usage else 0
            content = response.choices[0].message.content
            raw_response = {
                'id': response.id,
                'model': response.model,
                'content': content,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': tokens_input,
                    'completion_tokens': tokens_output,
                    'total_tokens': tokens_input + tokens_output
                }
            }

        elif provider == 'anthropic':
            # Anthropic has system prompt separate from messages
            # Filter out system message and use remaining as user/assistant turns
            non_system_messages = [m for m in messages if m['role'] != 'system']
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt or "",
                messages=non_system_messages,
                temperature=0
            )
            latency_ms = (time.time() - start_time) * 1000
            tokens_input = response.usage.input_tokens if response.usage else 0
            tokens_output = response.usage.output_tokens if response.usage else 0
            content = response.content[0].text
            raw_response = {
                'id': response.id,
                'model': response.model,
                'content': content,
                'finish_reason': response.stop_reason,
                'usage': {
                    'prompt_tokens': tokens_input,
                    'completion_tokens': tokens_output,
                    'total_tokens': tokens_input + tokens_output
                }
            }

        elif provider == 'google':
            # Google Gemini - combine system + messages into prompt
            import google.generativeai as genai
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            # Build conversation history
            history = []
            for m in messages:
                if m['role'] == 'system':
                    continue
                role = 'user' if m['role'] == 'user' else 'model'
                history.append({'role': role, 'parts': [m['content']]})

            # Last message is the query
            if history:
                query = history[-1]['parts'][0]
                chat_history = history[:-1] if len(history) > 1 else []
            else:
                query = ""
                chat_history = []

            chat = gen_model.start_chat(history=chat_history)
            response = chat.send_message(query)
            latency_ms = (time.time() - start_time) * 1000

            # Extract usage from response
            tokens_input = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            tokens_output = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            content = response.text
            raw_response = {
                'id': 'gemini-' + str(hash(content))[:8],
                'model': model,
                'content': content,
                'finish_reason': str(response.candidates[0].finish_reason) if response.candidates else 'unknown',
                'usage': {
                    'prompt_tokens': tokens_input,
                    'completion_tokens': tokens_output,
                    'total_tokens': tokens_input + tokens_output
                }
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Parse content (strip markdown code blocks if present)
        try:
            clean_content = content.strip()
            if clean_content.startswith('```'):
                # Remove markdown code block wrapper
                lines = clean_content.split('\n')
                # Remove first line (```json or ```) and last line (```)
                if lines[-1].strip() == '```':
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                clean_content = '\n'.join(lines)
            output = json.loads(clean_content)
            parsed_ok = True
        except json.JSONDecodeError:
            output = {}
            parsed_ok = False

        return {
            'output': output,
            'raw_response': raw_response,
            'http_status': http_status,
            'latency_ms': latency_ms,
            'tokens_input': tokens_input,
            'tokens_output': tokens_output,
            'parsed_ok': parsed_ok,
            'error': None
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        http_status = getattr(e, 'status_code', 500) if hasattr(e, 'status_code') else 500
        return {
            'output': {},
            'raw_response': None,
            'http_status': http_status,
            'latency_ms': latency_ms,
            'tokens_input': 0,
            'tokens_output': 0,
            'parsed_ok': False,
            'error': str(e)
        }


def run_inference(
    candidates_df: pd.DataFrame,
    model: str,
    provider: str,
    split: str,
    prompts: dict,
    max_repairs: int = 2,
    delay: float = 0.1
) -> tuple[list, list, list, list]:
    """
    Run inference on all candidates.

    Returns:
        results: prediction results
        log_entries: per-call log entries
        raw_responses: full API responses
        failures: failed extractions
    """
    # Initialize client
    if provider == 'openai':
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    elif provider == 'deepseek':
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
    elif provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    elif provider == 'google':
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
        client = None  # Google uses module-level configuration
    else:
        raise ValueError(f"Unknown provider: {provider}")

    results = []
    log_entries = []
    raw_responses = []
    failures = []

    for idx, row in candidates_df.iterrows():
        candidate_id = row['candidate_id']
        sentence = row['sentence']

        # Build messages with optional few-shot examples
        messages = [
            {"role": "system", "content": prompts['system']}
        ]

        # Add few-shot examples if available
        for example in prompts.get('fewshot_examples', []):
            messages.append({
                "role": "user",
                "content": prompts['user_template'].format(SENTENCE=example['sentence'])
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(example['output'])
            })

        # Add the actual query
        messages.append({
            "role": "user",
            "content": prompts['user_template'].format(SENTENCE=sentence)
        })

        # Initialize log entry with all spec fields
        log_entry = {
            'candidate_id': candidate_id,
            'split': split,
            'http_status': 200,
            'latency_ms': 0,
            'tokens_input': 0,
            'tokens_output': 0,
            'parsed_ok': True,
            'repair_used': False,
            'initial_parse_error': False,
            'was_repaired': False,
            'span_validation_error': False,
            'empty_content': False,
            'repair_attempts': 0
        }

        try:
            # Initial call
            api_result = call_api_with_timing(client, model, messages, prompts['schema'], provider, prompts['system'])

            # Update log entry
            log_entry['http_status'] = api_result['http_status']
            log_entry['latency_ms'] = api_result['latency_ms']
            log_entry['tokens_input'] = api_result['tokens_input']
            log_entry['tokens_output'] = api_result['tokens_output']
            log_entry['parsed_ok'] = api_result['parsed_ok']

            # Save raw response
            if api_result['raw_response']:
                raw_responses.append({
                    'candidate_id': candidate_id,
                    'attempt': 0,
                    **api_result['raw_response']
                })

            output = api_result['output']

            # Check for empty content
            if not output or (not output.get('rationale_short') and not output.get('cause_span')):
                log_entry['empty_content'] = True

            # Check for API error
            if api_result['error']:
                log_entry['initial_parse_error'] = True
                failures.append({
                    'candidate_id': candidate_id,
                    'attempt': 0,
                    'error_type': 'api_error',
                    'error_message': api_result['error']
                })
            elif not api_result['parsed_ok']:
                log_entry['initial_parse_error'] = True
                failures.append({
                    'candidate_id': candidate_id,
                    'attempt': 0,
                    'error_type': 'json_parse_error',
                    'error_message': 'Failed to parse JSON response'
                })
            else:
                # Validate output
                is_valid, error_msg = validate_output(output, sentence)

                if not is_valid:
                    log_entry['initial_parse_error'] = True
                    failures.append({
                        'candidate_id': candidate_id,
                        'attempt': 0,
                        'error_type': 'validation_error',
                        'error_message': error_msg
                    })

                    # Try repairs
                    for repair_attempt in range(max_repairs):
                        log_entry['repair_attempts'] += 1
                        messages.append({"role": "assistant", "content": json.dumps(output)})
                        messages.append({"role": "user", "content": prompts['repair'] + f"\nError: {error_msg}"})

                        repair_result = call_api_with_timing(client, model, messages, prompts['schema'], provider, prompts['system'])

                        # Update totals
                        log_entry['latency_ms'] += repair_result['latency_ms']
                        log_entry['tokens_input'] += repair_result['tokens_input']
                        log_entry['tokens_output'] += repair_result['tokens_output']

                        # Save raw response
                        if repair_result['raw_response']:
                            raw_responses.append({
                                'candidate_id': candidate_id,
                                'attempt': repair_attempt + 1,
                                **repair_result['raw_response']
                            })

                        output = repair_result['output']
                        is_valid, error_msg = validate_output(output, sentence)

                        if is_valid:
                            log_entry['was_repaired'] = True
                            log_entry['repair_used'] = True
                            break
                        else:
                            failures.append({
                                'candidate_id': candidate_id,
                                'attempt': repair_attempt + 1,
                                'error_type': 'validation_error',
                                'error_message': error_msg
                            })

                    if not is_valid:
                        log_entry['span_validation_error'] = True

            # Store result
            result = {
                'candidate_id': candidate_id,
                'attrib': output.get('attrib', False),
                'cause_span': output.get('cause_span'),
                'effect_span': output.get('effect_span'),
                'link_type': output.get('link_type', 'NO_CAUSAL_EXTRACTION'),
                'rationale_short': output.get('rationale_short', '')
            }
            results.append(result)
            log_entries.append(log_entry)

        except Exception as e:
            print(f"Error on {candidate_id}: {e}")
            log_entry['initial_parse_error'] = True
            log_entry['http_status'] = 500
            failures.append({
                'candidate_id': candidate_id,
                'attempt': 0,
                'error_type': 'exception',
                'error_message': str(e)
            })
            results.append({
                'candidate_id': candidate_id,
                'attrib': False,
                'cause_span': None,
                'effect_span': None,
                'link_type': 'NO_CAUSAL_EXTRACTION',
                'rationale_short': f'ERROR: {str(e)}'
            })
            log_entries.append(log_entry)

        # Rate limiting
        time.sleep(delay)

        # Progress
        if (len(results) % 50) == 0:
            print(f"  Processed {len(results)}/{len(candidates_df)}")

    return results, log_entries, raw_responses, failures


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=ANALYSIS_DIR
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def get_prompt_hash(prompts: dict) -> str:
    """Hash prompt content for versioning."""
    content = json.dumps(prompts, sort_keys=True)
    return hashlib.sha1(content.encode()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description='Run LLM baseline for DCHA-UNGD benchmark')
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., gpt-4o, gpt-4o-mini, deepseek-chat)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'deepseek', 'anthropic', 'google'],
                        help='API provider (default: openai)')
    parser.add_argument('--split', default='test', choices=['train', 'dev', 'test'],
                        help='Data split to run on (default: test)')
    parser.add_argument('--variant', default='zeroshot', choices=['zeroshot', 'fewshot'],
                        help='Prompt variant (default: zeroshot)')
    parser.add_argument('--run-name', help='Custom run name (default: auto-generated)')
    parser.add_argument('--max-repairs', type=int, default=2,
                        help='Max repair attempts per sample (default: 2)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between API calls in seconds (default: 0.1)')
    parser.add_argument('--limit', type=int, help='Limit number of candidates (for testing)')
    parser.add_argument('--data-dir', type=str,
                        help='Custom data directory (default: data/benchmark/dcha_ungd_extended_v1)')

    args = parser.parse_args()

    # Override DATA_DIR if custom path provided
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        if not DATA_DIR.is_absolute():
            DATA_DIR = ROOT_DIR / args.data_dir
        print(f"Using custom data directory: {DATA_DIR}")

    # Generate run name following spec: {DATE}_{provider}_{MODEL}_{variant}_v1
    if args.run_name:
        run_name = args.run_name
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        model_short = args.model.replace('-', '_')
        run_name = f"{date_str}_{args.provider}_{model_short}_{args.variant}_v1"

    # Save under provider subdirectory
    run_dir = RUNS_DIR / args.provider / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DCHA-UNGD LLM Baseline Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Split: {args.split}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {run_dir}")
    print("-" * 60)

    # Load data and prompts
    include_fewshot = (args.variant == 'fewshot')
    prompts = load_prompts(include_fewshot=include_fewshot)
    candidates_df = load_candidates(args.split)

    if args.limit:
        candidates_df = candidates_df.head(args.limit)

    print(f"Candidates to process: {len(candidates_df)}")
    print("-" * 60)

    # Run inference
    results, log_entries, raw_responses, failures = run_inference(
        candidates_df,
        model=args.model,
        provider=args.provider,
        split=args.split,
        prompts=prompts,
        max_repairs=args.max_repairs,
        delay=args.delay
    )

    # Save predictions as CSV
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(run_dir / 'predictions.csv', index=False)
    print(f"\nSaved predictions to {run_dir / 'predictions.csv'}")

    # Save predictions as JSONL
    with open(run_dir / 'predictions.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"Saved predictions to {run_dir / 'predictions.jsonl'}")

    # Save raw responses
    with open(run_dir / 'raw_responses.jsonl', 'w') as f:
        for resp in raw_responses:
            f.write(json.dumps(resp) + '\n')
    print(f"Saved raw responses to {run_dir / 'raw_responses.jsonl'}")

    # Save failures
    with open(run_dir / 'failures.jsonl', 'w') as f:
        for failure in failures:
            f.write(json.dumps(failure) + '\n')
    print(f"Saved failures to {run_dir / 'failures.jsonl'}")

    # Save run log
    with open(run_dir / 'run_log.jsonl', 'w') as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved run log to {run_dir / 'run_log.jsonl'}")

    # Save cost/latency CSV
    cost_latency_df = pd.DataFrame([{
        'candidate_id': e['candidate_id'],
        'split': e['split'],
        'http_status': e['http_status'],
        'latency_ms': e['latency_ms'],
        'tokens_input': e['tokens_input'],
        'tokens_output': e['tokens_output'],
        'parsed_ok': e['parsed_ok'],
        'repair_used': e['repair_used']
    } for e in log_entries])
    cost_latency_df.to_csv(run_dir / 'cost_latency.csv', index=False)
    print(f"Saved cost/latency to {run_dir / 'cost_latency.csv'}")

    # Compute reliability stats
    reliability_stats = {
        'total_candidates': len(log_entries),
        'invalid_json_count': sum(1 for e in log_entries if e['initial_parse_error']),
        'repair_count': sum(1 for e in log_entries if e['was_repaired']),
        'invalid_span_count': sum(1 for e in log_entries if e['span_validation_error']),
        'empty_content_count': sum(1 for e in log_entries if e['empty_content'])
    }

    # Calculate rates
    n = reliability_stats['total_candidates']
    if n > 0:
        reliability_stats['invalid_json_rate'] = reliability_stats['invalid_json_count'] / n
        reliability_stats['repair_rate'] = reliability_stats['repair_count'] / n
        reliability_stats['invalid_span_rate'] = reliability_stats['invalid_span_count'] / n
        reliability_stats['empty_content_rate'] = reliability_stats['empty_content_count'] / n

    # Calculate cost stats
    total_tokens_in = sum(e['tokens_input'] for e in log_entries)
    total_tokens_out = sum(e['tokens_output'] for e in log_entries)
    total_latency_ms = sum(e['latency_ms'] for e in log_entries)

    # Create manifest.yaml (as per KDD D&B spec)
    manifest = {
        'run_name': run_name,
        'provider': args.provider,
        'model': args.model,
        'api': 'chat.completions',
        'response_format': {
            'type': 'json_schema',
            'strict': True if args.provider == 'openai' else False
        },
        'temperature': 0,
        'max_tokens': 256,
        'prompt_version': 'v1',
        'prompt_version_hash': get_prompt_hash(prompts),
        'variant': args.variant,
        'fewshot': args.variant == 'fewshot',
        'split': args.split,
        'code_git_commit': get_git_commit(),
        'run_datetime_utc': datetime.utcnow().isoformat() + 'Z',
        'dataset_version': 'dcha_ungd_v1',
        'n_candidates': len(candidates_df),
        'max_repairs': args.max_repairs,
        'reliability_stats': reliability_stats,
        'usage_stats': {
            'total_tokens_input': total_tokens_in,
            'total_tokens_output': total_tokens_out,
            'total_tokens': total_tokens_in + total_tokens_out,
            'total_latency_ms': total_latency_ms,
            'avg_latency_ms': total_latency_ms / n if n > 0 else 0
        }
    }

    with open(run_dir / 'manifest.yaml', 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    print(f"Saved manifest to {run_dir / 'manifest.yaml'}")

    # Also save as JSON for backwards compatibility
    with open(run_dir / 'run_metadata.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved metadata to {run_dir / 'run_metadata.json'}")

    # Print summary
    stats = manifest['reliability_stats']
    usage = manifest['usage_stats']
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Total processed:   {stats['total_candidates']}")
    print(f"Invalid JSON:      {stats['invalid_json_count']} ({stats.get('invalid_json_rate', 0)*100:.1f}%)")
    print(f"Repaired:          {stats['repair_count']} ({stats.get('repair_rate', 0)*100:.1f}%)")
    print(f"Invalid spans:     {stats['invalid_span_count']} ({stats.get('invalid_span_rate', 0)*100:.1f}%)")
    print(f"Empty content:     {stats['empty_content_count']} ({stats.get('empty_content_rate', 0)*100:.1f}%)")
    print("-" * 60)
    print(f"Total tokens:      {usage['total_tokens']:,}")
    print(f"Avg latency:       {usage['avg_latency_ms']:.0f}ms")
    print("=" * 60)
    print(f"\nNext: Run evaluation with:")
    print(f"  python eval/evaluate.py {run_name} --split {args.split} --separate-csvs")


if __name__ == '__main__':
    main()

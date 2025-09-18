import os
import argparse
from mvsep_client import MVSEPClient


def main():
    parser = argparse.ArgumentParser(description="Example: submit file to MVSEP, wait and download.")
    parser.add_argument('--token', required=True, help='API token')
    parser.add_argument('--input', required=True, help='Path to input audio file')
    parser.add_argument('--output', default='./mvsep_out', help='Directory to store results')
    parser.add_argument('--sep_type', type=int, default=40, help='Separation type id')
    parser.add_argument('--add_opt1', default=81, help='Additional option 1')
    parser.add_argument('--add_opt2', default=None, help='Additional option 2')
    parser.add_argument('--add_opt3', default=None, help='Additional option 3')
    parser.add_argument('--poll', type=int, default=10, help='Poll interval (seconds)')
    parser.add_argument('--timeout', type=int, default=60*30, help='Timeout (seconds)')
    args = parser.parse_args()

    client = MVSEPClient(api_key=args.token, debug=True)

    print('Submitting file...')
    task_hash = client.submit_file(
        file_path=args.input,
        sep_type=args.sep_type,
        add_opt1=args.add_opt1,
        add_opt2=args.add_opt2,
        add_opt3=args.add_opt3,
    )
    print(f'Submitted, hash: {task_hash}')

    print('Waiting for completion...')
    status = client.wait_for_done(task_hash, poll_interval=args.poll, timeout=args.timeout)
    print(f'Final status: {status.get("status")}')

    print('Downloading results...')
    os.makedirs(args.output, exist_ok=True)
    client.download_result(status, args.output)
    print('Download complete.')


if __name__ == '__main__':
    main()

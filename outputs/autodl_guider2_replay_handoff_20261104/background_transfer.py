"""Start a durable transfer; the password exists only in process memory."""
import getpass
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
password = getpass.getpass('SSH password (not saved): ')
environment = dict(os.environ, SSHPASS=password)
with (here / 'transfer.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(['sshpass', '-e', 'bash', str(here / 'transfer_from_source.sh')],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, start_new_session=True)
(here / 'transfer.pid').write_text(str(process.pid) + '\n')
print('Background transfer PID:', process.pid)
print('Log:', here / 'transfer.log')
print('Completion marker:', here / 'transfer.complete')

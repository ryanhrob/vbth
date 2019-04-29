# Voice Biometrics Test Harness

## Requirements
* Python 3.7.x
* Tested only on Windows 10 but likely to work on Linux, MacOS, etc too

## Usage
* The script contains subcommands which take options (much like git).
* To see the list of commands run "python vbth.py help"
* To see the options for a given command run "python vbth.py <command> --help"

Example commands:
* python vbth.py runtest --corpus speakers --csvout results.csv --impl fast
* python vbth.py analyse --results results.csv --th_user 7

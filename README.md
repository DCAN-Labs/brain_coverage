# Brain Coverage

The purpose of these scripts is to take the BOLD images created by the abcd-hcp-pipeline and the 2mm MNI T1 mask and calculate the percentage of overlap, then add that value to the participants.tsv. 

The Juptyer notebook was the original collection of commands created by Anders Perrone.

brain_coverage.py is the python script that rae McCollum created based on the Juptyer commands.

- In order to cut down on calculation time, this script is designed to take in a subset of ABCC subjects from the NGDR using wildcards. Since every subject ID starts with sub-NDARINV and the next value is selected from 0-9 then A-Z, you would enter /path/to/folders/sub-NDARINVA* to select the subset of IDs that follow this pattern. 

- For running one subset of subjects (using the wildcards), which is around 600 subjects, I give at least 30 hours of time and 200 GB of memory.

advanced_brain_coverage.py is a python script that Greg Conan developed upon from brain_coverage.py. 

- IMHO, this script is a bit overcomplicated and hard to follow but I think the more function based implementation is better than the original brain_coverage.py that's just a massive nested for loop.

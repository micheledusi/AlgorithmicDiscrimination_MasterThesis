#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Reading Jonathan Neidel's Job Titles Dataset
# The dataset contains about 73k job titles. For simplicity purposes, we use only the tiles
# composed by a single word.

import settings


JOBS_TITLES_FILE = settings.FOLDER_DATA + "/jneidel/job-titles.txt"
ONEWORD_JOBS_FILE = settings.FOLDER_DATA + "/jneidel/oneword-job-titles.txt"


def read_jobs(filepath: str) -> list[str]:
	jobs: list[str] = []
	with open(filepath, "r") as infile:
		for j in infile:
			jobs.append(j.strip())
	return jobs


ALL_OCCUPATIONS: list[str] = read_jobs(JOBS_TITLES_FILE)
ONEWORD_OCCUPATIONS: list[str] = read_jobs(ONEWORD_JOBS_FILE)


AN_PREFIXES: tuple[str, ...] = ('a', 'e', 'i', 'o')
A_PREFIXES: tuple[str, ...] = ('ow', 'uni')


def infer_indefinite_article(job_word: str) -> str:
	"""
	Returns the indefinite article for the give noun based on simplistic assumptions.
	:param job_word: A give noun
	:return: The corresponding indefinite article ('a'/'an'), based on some assumptions
	"""
	if job_word.startswith(AN_PREFIXES) and not job_word.startswith(A_PREFIXES):
		return 'an'
	return 'a'


if __name__ == '__main__':
	jobs_list: list[str] = []
	with open('../../' + JOBS_TITLES_FILE, "r") as f_in:
		for job in f_in:
			job = job.strip()
			if ' ' not in job:
				jobs_list.append(job)

	print("A total number of {} one-word jobs was found in the dataset.".format(len(jobs_list)))
	# Writing the one-word jobs in a separate file
	with open('../../' + ONEWORD_JOBS_FILE, "w") as f_out:
		print('\n'.join(jobs_list), file=f_out)


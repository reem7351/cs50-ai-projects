import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}

    pages = list(corpus.keys())
    total_pages = len(pages)

    links = corpus[page]
    num_links = len(links)

    # If page has no links, treat it as linking to all pages
    if num_links == 0:
        for p in pages:
            distribution[p] = 1.0 / total_pages
        return distribution

    # Base probability for all pages
    for p in pages:
        distribution[p] = (1 - damping_factor) / total_pages

    # Add damping_factor probability to linked pages
    for link in links:
        distribution[link] += damping_factor / num_links

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
        # Initialize counts for each page
    pagerank = {page: 0 for page in corpus}

    # Choose a starting page at random
    page = random.choice(list(corpus.keys()))

    # Sample n pages
    for _ in range(n):
        pagerank[page] += 1

        # Get next page based on transition model
        probs = transition_model(corpus, page, damping_factor)

        r = random.random()
        cumulative = 0.0
        for p, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                page = p
                break

    # Convert counts to probabilities
    for p in pagerank:
        pagerank[p] /= n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)

    # Start with equal rank for every page
    ranks = {page: 1 / N for page in corpus}

    while True:
        new_ranks = {}

        for page in corpus:
            total = 0.0

            for possible_page in corpus:
                links = corpus[possible_page]

                # If a page has no links, treat it as linking to all pages
                if len(links) == 0:
                    total += ranks[possible_page] / N
                elif page in links:
                    total += ranks[possible_page] / len(links)

            new_ranks[page] = (1 - damping_factor) / N + damping_factor * total

        # Check convergence
        converged = True
        for page in ranks:
            if abs(new_ranks[page] - ranks[page]) > 0.001:
                converged = False
                break

        ranks = new_ranks

        if converged:
            break

    # Normalize to sum to 1 (tiny float safety)
    s = sum(ranks.values())
    for page in ranks:
        ranks[page] /= s

    return ranks


if __name__ == "__main__":
    main()

# Pokémon TCG AI Battle Challenge (Simulation Track)

## Description
This competition is the **Simulation competition** of the Pokémon TCG AI Battle Challenge. 

As a strategic game of its own, Pokémon TCG players must make gameplay decisions while being mindful of their opponent's own strategies, decks, and hands. When formulating their approach to the game, players need to take into consideration various Pokémon types and thousands of different card combinations to consider various gameplay possibilities. In addition, other factors such as card draws and coin tosses introduce additional gameplay variables. The AI Training Agent’s training will be conducted within this competitive simulation framework.

Not knowing what cards an opponent holds presents a core challenge for an AI Training Agent. Participants are encouraged to explore novel methodologies for strategy learning and decision making in this dynamic environment.

Participants will be provided with a simulator (SDK) for training and testing. This toolkit uses the same logic as the Kaggle competition environment, making it suitable for local debugging and reinforcement learning.

Using rule-based programming alone may not ensure a high ranking. Winning a Pokémon TCG game requires forward thinking, real-time adaptation, and optimal decision-making. With many different deck variations available and evolving strategies, no two games are alike. The AI Training Agent must demonstrate high analytical capacity, adaptability, and be ready for the unexpected.

---

## Evaluation
Each day your team is able to submit up to **5 agents** to the competition. Each submission will play Episodes (games) against other agents on the ladder that have a similar skill rating. Over time skill ratings will go up with wins or down with losses and evened out with ties. To reduce the number of agents playing and increase the number of episodes each team participates in, we only track the **latest 2 submissions** and use those for final submissions.

Every agent submitted will continue to play episodes until the end of the competition, with newer agents playing a much more frequent number of episodes. On the leaderboard only your best scoring agent will be shown, but you can track the progress of all of your submissions on your Submissions page.

Each Submission has an estimated Skill Rating which is modeled by a Gaussian $N(\mu, \sigma^2)$ where $\mu$ is the estimated skill and $\sigma$ represents the uncertainty of that estimate which will decrease over time.

When you upload a Submission, we first play a **Validation Episode** where that Submission plays against copies of itself to make sure it works properly. If the Episode fails, the Submission is marked as Error and you can download the agent logs to help figure out why. Otherwise, we initialize the Submission with $\mu_0=600$ and it joins the pool of All Submissions for ongoing evaluation.

We repeatedly run Episodes from the pool of All Submissions, and try to pick Submissions with similar ratings for fair matches. Newly submitted agents will be given an increased rate in the number of episodes run to give you faster feedback.

### Ranking System
After an Episode finishes, we'll update the Rating estimate for all Submissions in that Episode. If one Submission won, we'll increase its $\mu$ and decrease its opponent's $\mu$ -- if the result was a draw, then we'll move the two $\mu$ values closer towards their mean. The updates will have magnitude relative to the deviation from the expected result based on the previous $\mu$ values, and also relative to each Submission's uncertainty $\sigma$. We also reduce the $\sigma$ terms relative to the amount of information gained by the result. The score by which your agent wins or loses an Episode does not affect the skill rating updates.

### Final Evaluation
At the submission deadline on August 16, 2026, additional submissions will be locked. From August 16, 2026 for approximately two weeks, we will continue to run games. At the conclusion of this period, the leaderboard is final.

---

## Timeline
* **June 16, 2026 11:00 am UTC** - Start Date.
* **August 9, 2026** - Entry Deadline. You must accept the competition rules before this date in order to compete.
* **August 9, 2026** - Team Merger Deadline. This is the last day participants may join or merge teams.
* **August 16, 2026** - Final Submission Deadline.
* **August 17, 2026 to (approx.) August 31, 2026** - We will continue to run games, or until the leaderboard has reached convergence. At the conclusion of this period, the leaderboard is final.

*All deadlines are at 11:59 PM UTC on the corresponding day.*

---

## Prizes
The Competition track itself does not include monetary prizes. However, participants who submit a report to the Hackathon track will be eligible for prize awards. Final rankings for Hackathon prizes will be determined based on both the Competition leaderboard performance and the Hackathon evaluation.

---

## How to Play Pokémon TCG
* Download the most recent rulebook for the Pokémon Trading Card Game.
* View the Competition Data Page for more information on cards and decks available for this tournament.
* Get information on the Pokémon TCG, the Play! Pokémon program and more on The Pokémon Company's Rules & Resources page.

---

## Simulator API Documentation
Battles are run on the **cabt Engine**, a Pokémon TCG battle simulator built for `kaggle-environments`.

Each turn, your agent receives an observation — including game logs, the current board state, and a list of legal options — and returns the indices of the options it selects. The engine only ever presents legal moves.

* **API Documentation**: [matsuoinstitute.github.io/cabt/](https://matsuoinstitute.github.io/cabt/)
* *Note: There are a few differences between the official Pokémon TCG rules and the simulator behavior, which can be found in the documentation.*

---

## How to Submit to this Competition
Submissions need to be a `.tar.gz` bundle with `main.py` at the top level directory (not nested) and include a `deck.csv`.

* To create a submission, bundle the files using:
  ```bash
  tar -czvf submission.tar.gz *.py deck.csv
  ```
* Upload this under the **My Submissions** tab.
* Your submission will start with a scheduled game vs itself to ensure everything is working before being entered into the matchmaking pool.

### Environment & Resources
* **File Directory:** Your files will be located in `/kaggle_simulations/agent/`. Ensure all your file imports are set appropriately (e.g., using relative paths or prepending `/kaggle_simulations/agent/` to path searches).
* **Submission Size Limit:** 197.7 MiB (Max)
* **HDD Space:** 11.8 GiB
* **RAM:** 12.2 GiB
* **vCPUs:** 2
* **Docker Image:** Uses the standard `kaggle-environments` 1.14.10 image.

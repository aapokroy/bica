import random

import graphviz as gv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.contrib.concurrent import process_map

from lib.schemas import Action, Relationship

N_INTENTIONALITIES = 3
VALENCE_IDX = 0
AROUSAL_IDX = 1
DOMINANCE_IDX = 2


class Community:
    def __init__(
        self,
        n_agents: int,
        arousal_bias: float,
        temperature: float,
        appraisal_change_rate: float,
        feelings_inertia: float,
        appraisal_decay: float,
        max_actions_per_turn: int,
        relationships: list[Relationship],
        actions: list[Action],
    ) -> None:
        self.n_agents = n_agents
        self._arousal_bias = arousal_bias
        self._temperature = temperature
        self._appraisal_change_rate = appraisal_change_rate
        self._feelings_inertia = feelings_inertia
        self._appraisal_decay = appraisal_decay
        self._max_actions_per_turn = max_actions_per_turn

        self._appraisals = np.zeros((n_agents, n_agents, N_INTENTIONALITIES))
        self._feelings = np.zeros((n_agents, n_agents, N_INTENTIONALITIES))
        self._relationships = np.full((n_agents, n_agents), fill_value=None, dtype=object)

        self._relationship_pool = relationships
        self._action_pool = actions

    def add_agent(self) -> None:
        n_agents = self.n_agents + 1
        appraisals = np.zeros((n_agents, n_agents, N_INTENTIONALITIES))
        feelings = np.zeros((n_agents, n_agents, N_INTENTIONALITIES))
        relationships = np.full((n_agents, n_agents), fill_value=None, dtype=object)

        appraisals[:self.n_agents, :self.n_agents] = self._appraisals
        feelings[:self.n_agents, :self.n_agents] = self._feelings
        relationships[:self.n_agents, :self.n_agents] = self._relationships

        self.n_agents = n_agents
        self._appraisals = appraisals
        self._feelings = feelings
        self._relationships = relationships

    def _find_closest_relationship(self, feeling: np.ndarray) -> Relationship:
        """Find the closest relationship centroid to the given feeling vector."""
        return min(
            self._relationship_pool,
            key=lambda x: np.linalg.norm(feeling - np.array(x.feeling_centroid)),
        )

    def _update_relationships(self) -> None:
        """Update agent relationships based on the current feelings."""
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                self._relationships[i, j] = self._find_closest_relationship(self._feelings[i, j])

    def _evaluate_action(self, actor: int, recipient: int, action: Action) -> float:
        """Evaluate the given action for the actor-recipient pair."""
        apraisal = self._appraisals[actor, recipient]

        # Base evaluation (cosine similarity multiplied by norm of the appraisal change)
        evaluation = np.array(action.actor_appraisal_delta) @ apraisal.T
        evaluation /= np.linalg.norm(apraisal) or 1.0
        
        # Actor will more likely select recipient with higher arousal
        evaluation += self._arousal_bias * apraisal[AROUSAL_IDX]
        
        return evaluation

    def _get_all_possible_actions(self, actor: int) -> list[tuple[int, int, Action, float]]:
        possible_actions: list[tuple[int, int, Action, float]] = []
        for recipient in range(self.n_agents):
            if actor == recipient:
                continue
            for action in self._action_pool:
                if all([
                    self._relationships[actor, recipient] in action.actor_relationships,
                    self._relationships[recipient, actor] in action.recipient_relationships,
                ]):
                    evaluation = self._evaluate_action(actor, recipient, action)
                    if evaluation >= 0:
                        proba = np.exp(evaluation / self._temperature)
                        possible_actions.append((actor, recipient, action, proba))
        return possible_actions
        

    def turn(self) -> list[tuple[int, int, Action]]:
        self._update_relationships()

        possible_actions_chunked = process_map(
            self._get_all_possible_actions,
            range(self.n_agents),
            max_workers=16,
        )
        possible_actions = [action for chunk in possible_actions_chunked for action in chunk]

        agent2cluster: dict[int, int] = {}
        cluster2agents: dict[int, set[int]] = {}
        actions: list[tuple[int, int, Action]] = []
        action_counts = {agent: 0 for agent in range(self.n_agents)}
        while possible_actions:
            probas = np.array([proba for _, _, _, proba in possible_actions])
            probas /= probas.sum()

            idx = np.random.choice(range(len(possible_actions)), p=probas)
            actor, recipient, action, _ = possible_actions[idx]
            actions.append((actor, recipient, action))
            action_counts[actor] += 1
            action_counts[recipient] += 1
            if action_counts[actor] >= self._max_actions_per_turn:
                possible_actions = [
                    (actor_, recipient_, action_, proba_)
                    for actor_, recipient_, action_, proba_ in possible_actions
                    if actor_ != actor
                ]
            if action_counts[recipient] >= self._max_actions_per_turn:
                possible_actions = [
                    (actor_, recipient_, action_, proba_)
                    for actor_, recipient_, action_, proba_ in possible_actions
                    if recipient_ != recipient
                ]
            possible_actions = [
                (actor_, recipient_, action_, proba_)
                for actor_, recipient_, action_, proba_ in possible_actions
                if (actor_, recipient_) != (actor, recipient)
            ]

            if recipient in agent2cluster:
                cluster = agent2cluster[recipient]
                if actor not in cluster2agents[cluster]:
                    cluster2agents[cluster].add(actor)
                    agent2cluster[actor] = cluster
            else:
                cluster = len(cluster2agents)
                agent2cluster[actor] = cluster
                agent2cluster[recipient] = cluster
                cluster2agents[cluster] = {actor, recipient}

            possible_actions = [
                (actor, recipient, action, proba)
                for actor, recipient, action, proba in possible_actions
                if actor not in agent2cluster or recipient in cluster2agents[agent2cluster[actor]]
            ]

        
        # Tanh transformation is used to keep the appraisals in the valid range
        # and make the change rate slower when the value is close to the limits
        transformed_appriasals = self._appraisals.copy()
        transformed_appriasals = np.arctanh(self._appraisals)

        for actor, recipient, action in actions:
            # The change rate is increased when the dominance is higher, which means
            # that one agent has more influence over the other
            change_rate = self._appraisal_change_rate 
            change_rate *= (1 + self._appraisals[actor, recipient, DOMINANCE_IDX])
            delta = np.array(action.actor_appraisal_delta)
            transformed_appriasals[actor, recipient] += change_rate * delta

            change_rate = self._appraisal_change_rate 
            change_rate *= (1 + self._appraisals[recipient, actor, DOMINANCE_IDX])
            delta = np.array(action.recipient_appraisal_delta)
            transformed_appriasals[recipient, actor] += change_rate * delta

            # Other agents valence is also affected by the action. The change rate in this case
            # is multiplied by agent's valence towards the recipient. So, if an agent has a positive
            # valence towards the recipient, agent's appraisal will be changed in the same direction
            # as the recipient's appraisal, but with a smaller magnitude. And if the valence is negative,
            # the change will be in the opposite direction. This is done to simulate the effect of
            # observing the interaction between the actor and the recipient.

            cluster = agent2cluster[actor]
            possible_observers = list(set(cluster2agents[cluster]) - {actor, recipient})
            if possible_observers:
                observer = np.random.choice(possible_observers)

                change_rate = self._appraisal_change_rate
                change_rate *= (1 + self._appraisals[observer, actor, DOMINANCE_IDX])
                change_rate *= self._appraisals[observer, recipient, VALENCE_IDX]
                delta = np.array(action.recipient_appraisal_delta[VALENCE_IDX])
                transformed_appriasals[observer, actor, VALENCE_IDX] += change_rate * delta

                change_rate = self._appraisal_change_rate
                change_rate *= (1 + self._appraisals[observer, recipient, DOMINANCE_IDX])
                change_rate *= self._appraisals[observer, actor, VALENCE_IDX]
                delta = np.array(action.actor_appraisal_delta[VALENCE_IDX])
                transformed_appriasals[observer, recipient, VALENCE_IDX] += change_rate * delta

        appraisals = np.tanh(transformed_appriasals)
        appraisals[:, :, AROUSAL_IDX] = np.clip(appraisals[:, :, AROUSAL_IDX], 0, 1)
        self._appraisals = appraisals

        # Feelings are calculated as the exponential moving average of the appraisals
        self._feelings = self._feelings * self._feelings_inertia + self._appraisals * (1 - self._feelings_inertia)

        self._appraisals *= 1 - self._appraisal_decay

        return actions

    def plot_relationships(self) -> None:
        relationships = [rel for rel in self._relationship_pool if rel.title != 'Neutral']
        rel_to_int = {rel.title: i for i, rel in enumerate(relationships)}
        int_to_rel = {i: title for title, i in rel_to_int.items()}

        matrix = np.array([
            [
                rel_to_int.get(self._relationships[i, j].title if self._relationships[i, j] else None, np.nan)
                for j in range(self.n_agents)
            ] for i in range(self.n_agents)
        ])

        bounds = [-0.5 + i for i in range(len(int_to_rel) + 1)]
        norm = mcolors.BoundaryNorm(bounds, len(int_to_rel))

        cbar = plt.imshow(matrix, cmap='tab20', norm=norm)
        cbar = plt.colorbar(cbar, ticks=range(len(int_to_rel)))
        cbar.ax.set_yticklabels([int_to_rel[i] for i in range(len(int_to_rel))])
        plt.title('Relationship map')

    def run(self, n_turns: int, plot_interval: int = 10):
        action_counts = {action.title: [] for action in self._action_pool}
        activity_map = np.zeros((self.n_agents, self.n_agents))
        relationship_pairs = np.zeros((len(self._relationship_pool), len(self._relationship_pool)))
        for turn in range(n_turns):
            actions = self.turn()

            for action in self._action_pool:
                action_counts[action.title].append(0)
            for actor, recipient, action in actions:
                action_counts[action.title][-1] += 1
                activity_map[actor, recipient] = 0.9 * activity_map[actor, recipient] + 0.1
            activity_map *= 0.95

            rel_to_int = {rel.title: i for i, rel in enumerate(self._relationship_pool)}
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    x, y = (
                        rel_to_int[self._relationships[i, j].title],
                        rel_to_int[self._relationships[j, i].title],
                    )
                    relationship_pairs[x, y] = 0.9 * relationship_pairs[x, y] + 0.1
            relationship_pairs *= 0.95

            if turn % plot_interval == 0:
                fig = plt.figure(figsize=(10, 14))
                plt.subplot(4, 3, 4)
                self.plot_relationships()

                plt.subplot(4, 3, 5)
                plt.imshow(activity_map)
                plt.title('Activity map')

                plt.subplot(4, 3, 6)
                plt.imshow(relationship_pairs)
                plt.title('Relationship pairs')
                plt.xticks(list(rel_to_int.values()), list(rel_to_int.keys()), rotation=90)
                plt.yticks(list(rel_to_int.values()), list(rel_to_int.keys()))

                plt.subplot(4, 1, 1)
                plt.stackplot(range(len(action_counts[next(iter(action_counts))])), action_counts.values())
                plt.legend(action_counts.keys(), loc='upper left', framealpha=0.8, frameon=True)
                plt.xlabel('Turn')
                plt.title('Action counts')


                relationship_to_color = {
                    rel.id: mcolors.to_hex(mcolors.hsv_to_rgb((i / len(self._relationship_pool), 1, 1)))
                    for i, rel in enumerate(self._relationship_pool)
                }

                dot = gv.Digraph()

                dot.attr('graph', concentrate='false')
                dot.attr('edge', arrowhead='vee')
                dot.attr('edge', arrowsize='0.5')
                dot.attr('node', shape='circle')
                dot.graph_attr['nodesep'] = '0.1'
                dot.graph_attr['ranksep'] = '0.1'
                dot.graph_attr['size'] = '10,10'
                dot.attr('edge', penwidth='2')
                # dot.attr('node', label='')

                for agent in range(self.n_agents):
                    dot.node(str(agent))
                for i in range(self.n_agents):
                    for j in range(self.n_agents):
                        if i == j:
                            continue
                        rel = self._relationships[i, j]
                        if rel is not None and sum(map(abs, rel.feeling_centroid)) != 0:
                            color = relationship_to_color[self._relationships[i, j].id]
                            dot.edge(str(i), str(j), color=color)

                plt.subplot(2, 1, 2)
                dot.render('graph', format='png', cleanup=True)
                plt.imshow(plt.imread('graph.png'))
                plt.axis('off')

                ax = plt.subplot(4, 1, 3)
                for rel in self._relationship_pool:
                    plt.plot(0, 0, color=relationship_to_color[rel.id], label=rel.title)
                plt.axis('off')
                plt.legend(loc='upper left', framealpha=0.8, frameon=True)

                fig.tight_layout()
                yield fig

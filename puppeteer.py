import abc
from typing import List, Dict, Tuple, Type

import re
import string
import spacy
import numpy as np
import matplotlib.pyplot as plt

from .agenda import Action, Agenda, AgendaState
from .logging import Logger
from .observation import Observation, IntentObservation
from .extractions import Extractions


class PuppeteerPolicy(abc.ABC):
    """Handles inter-agenda decisions about behavior.

    An PuppeteerPolicy is responsible for making decisions about what agenda to run and when to restart agendas, based
    on agenda states. Agenda-level decisions, e.g., which of the agenda's actions to choose in a given situation, are
    delegated to the AgendaPolicy instance associated with each Agenda.

    This class is an abstract class defining all methods that a PuppeteerPolicy must implement, most notably the act()
    method.

    A concrete PuppeteerPolicy subclass instance (object) is tied to a Puppeteer instance and is responsible for all
    action decisions for the conversation handled by the Puppeteer instance. This means that a PuppeteerPolicy instance
    is tied to a specific conversation, and may hold conversation-specific state.
    """

    def __init__(self, agendas: List[Agenda]) -> None:
        """Initialize a new policy

        Args:
            agendas: The agendas used by the policy.
        """
        self._agendas = agendas

    @abc.abstractmethod
    def act(self, agenda_states: Dict[str, AgendaState], extractions: Extractions) -> List[Action]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """
        raise NotImplementedError()
'''
    @abc.abstractmethod
    def plot_state(self, fig: plt.Figure, agenda_states: Dict[str, AgendaState]) -> None:
        raise NotImplementedError()
'''

class DefaultPuppeteerPolicy(PuppeteerPolicy):
    """Handles inter-agenda decisions about behavior.

    This is the default PuppeteerPolicy implementation. See PuppeteerPolicy documentation for further details.
    """

    def __init__(self, agendas: List[Agenda]) -> None:
        """Initialize a new policy

        Args:
            agendas: The agendas used by the policy.
        """
        super(DefaultPuppeteerPolicy, self).__init__(agendas)
        # State
        # self._current_agenda = None
        self._active_agendas: Dict[str, Agenda] = {} # agendas that are able to start
        self._kicked_off_agendas: Dict[str, Agenda] = {} # agendas that have already been kicked off
        self._turns_without_progress = {a.name: -1 for a in agendas} #number of consecutive turns without progress
        self._times_made_current = {a.name: 0 for a in agendas}
        self._action_history: Dict[str, List[Action]] = {a.name: [] for a in agendas}
        self._log = Logger()

    def act(self, agenda_states: Dict[str, AgendaState], extractions: Extractions) -> List[Action]:
        """"Picks zero or more appropriate actions to take, given the current state of the conversation.

        See documentation of this method in PuppeteerPolicy.

        Args:
            agenda_states: For each agenda (indexed by name), the AgendaState object holding the current belief about
                the state of the agenda, based on the latest observations -- the observations that this method is
                reacting to.

        Returns:
            A list of Action objects representing actions to take, in given order.
        """

        if len(self._active_agendas) == 0:
            # if no active agendas ==> then do nothing
            self._log.add("No active agenda, will do nothing.")
            return []

        self._log.begin("Kicked-off agendas")

        progress_flag = {} #binary value whether or not an agenda makes progress at that turn
        if len(self._kicked_off_agendas) == 0:
            # if a list of kicked-off agendas is empty but there are active agendas ==> kick off one of them
            for agenda_name, agenda in self._active_agendas.items():
                if self._times_made_current[agenda_name] == 0:
                    # Kick off this agenda for the first time.
                    self._log.add("We will kick off {} for the first time.".format(agenda_name))
                    self._kicked_off_agendas[agenda_name] = agenda
                    self._times_made_current[agenda_name] += 1
                    progress_flag[agenda_name] = True
                    break

        if extractions.has_extraction("kickoff"):
            # When one agenda reached the terminus state and kick off another active agenda
            kicked_off_agenda_name = extractions.extraction("kickoff")["kickoff_agenda"]
            self._log.add("We will kick off {} for the first time.".format(kicked_off_agenda_name))
            self._kicked_off_agendas[kicked_off_agenda_name] = self._active_agendas[kicked_off_agenda_name]
            self._times_made_current[kicked_off_agenda_name] += 1
            progress_flag[kicked_off_agenda_name] = True

        for agenda_name, agenda in self._kicked_off_agendas.items():
            # Check progress only if the agenda has kicked off
            agenda_state = agenda_states[agenda_name]
            progress_flag[agenda_name] = agenda.policy.made_progress(agenda_state) \
                if agenda_name not in progress_flag else progress_flag[agenda_name]
            if progress_flag[agenda_name] == True:
                self._log.add("We have made progress with {}.".format(agenda_name))
                # reset turns_without_progress if making progress.
                self._turns_without_progress[agenda_name] = 0
            else:
                self._log.add("We have not made progress with {}.".format(agenda_name))
                # increase turns_without_progress by 1 if not.
                self._turns_without_progress[agenda_name] += 1

        # if all(value >= 2 for value in self._turns_without_progress.values()):
        if all(self._turns_without_progress[agenda_name] >= 2 for agenda_name in self._kicked_off_agendas):
            # if each of kicked-off agendas has been idle for 2 consecutive turns or more and
            # there are other active agendas ==> kick off one of them
            self._log.add("Each of kicked-off agendas have been idle for 2 consecutive turns or more.")
            self._log.add("Finding other agenda that is active but not kicked-off.")
            for agenda_name, agenda in self._active_agendas.items():
                if self._times_made_current[agenda_name] == 0:
                    # Kick off this agenda for the first time.
                    self._log.add("We will kick off {} for the first time.".format(agenda_name))
                    self._kicked_off_agendas[agenda_name] = agenda
                    self._times_made_current[agenda_name] += 1
                    progress_flag[agenda_name] = True
                    self._log.add("We have made progress with {}.".format(agenda_name))
                    # reset turns_without_progress if making progress.
                    self._turns_without_progress[agenda_name] = 0
                    break

        self._log.end()

        self._log.begin("Active agendas")

        for agenda_name, agenda in self._active_agendas.items():
            # Report active agendas and their metadata such as turns_without_progress, times_made_current, action_history
            self._log.begin("{}".format(agenda_name))

            agenda_state = agenda_states[agenda_name]
            self._log.add("Turns without progress: {}".format(self._turns_without_progress[agenda_name]))
            self._log.add("Times used: {}".format(self._times_made_current[agenda_name]))
            self._log.add("Action history: {}".format([a.name for a in self._action_history[agenda_name]]))

            self._log.end()

        self._log.end()

        finished_agenda_names: List[str] = []
        actions: List[Action] = []

        # print("active_agendas", list(self._active_agendas.keys()))
        # print("kicked_off_agendas", list(self._kicked_off_agendas.keys()))
        # print("times_made_current", self._times_made_current)
        # print("turns_without_progress", self._turns_without_progress)
        # print("progress_flag", progress_flag)

        for agenda_name, agenda in self._kicked_off_agendas.items():
            # This agenda has been activated and kicked off
            # Run and see if we get some actions.
            # print(agenda_name)

            agenda_state = agenda_states[agenda_name]
            done = progress_flag[agenda_name] and agenda.policy.is_done(agenda_state)
            turns_without_progress = self._turns_without_progress[agenda_name]
            action_history = self._action_history[agenda_name]

            self._log.begin("Picking actions for {}.".format(agenda_name))
            actions += agenda.policy.pick_actions(agenda_state, action_history, turns_without_progress, extractions)
            # for act in actions:
            #     print(agenda_name, act.text)
            self._log.end()
            self._action_history[agenda.name].extend(actions)

            if done:
                # We reach the terminus state of this agenda.
                # Will continue on other active agendas.
                self._log.add("{} is in a terminal state, so it will be stopped.".format(agenda_name))
                agenda_state.reset()
                finished_agenda_names.append(agenda_name)

        # Remove finished agendas from a list of active agendas.
        for agenda_name in finished_agenda_names:
            del self._active_agendas[agenda_name]
            del self._kicked_off_agendas[agenda_name]
            self._turns_without_progress[agenda_name] = -1

        return actions


    # agenda = self._current_agenda
    # last_agenda = None
    # actions: List[Action] = []

    # if agenda is not None:
    #     self._log.begin(f"Current agenda is {agenda.name}.")
    #     agenda_state = agenda_states[agenda.name]

    #     self._log.begin(f"Puppeteer policy state for {agenda.name}:")
    #     self._log.add(f"Turns without progress: {self._turns_without_progress[agenda.name]}")
    #     self._log.add(f"Times used: {self._times_made_current[agenda.name]}")
    #     self._log.add(f"Action history: {[a.name for a in self._action_history[agenda.name]]}")
    #     self._log.end()

    #     # Update agenda state based on message.
    #     # What to handle in output?
    #     progress_flag = agenda.policy.made_progress(agenda_state)
    #     # Note that even if the agenda is considered done at this point, having reached a
    #     # terminal state as the result of the incoming observations, it still gets to do
    #     # a final action.
    #     done_flag = progress_flag and agenda.policy.is_done(agenda_state)
    #     if progress_flag:
    #         self._log.add("We have made progress with the agenda.")
    #         self._turns_without_progress[agenda.name] = 0
    #     else:
    #         self._log.add("We have not made progress with the agenda.")
    #         # At this point, the current agenda (if there is
    #         # one) was the one responsible for our previous
    #         # reply in this convo. Only this agenda has its
    #         # turns_without_progress counter incremented.
    #         self._turns_without_progress[agenda.name] += 1

    #     turns_without_progress = self._turns_without_progress[agenda.name]

    #     if turns_without_progress >= 2:
    #         self._log.add("The agenda has been going on for too long without progress and will be stopped.")
    #         agenda_state.reset()
    #         self._current_agenda = None
    #         del self._active_agendas[agenda.name] #remove current agenda from the active list
    #         last_agenda = agenda
    #     else:
    #         # Run and see if we get some actions.
    #         action_history = self._action_history[agenda.name]
    #         self._log.begin("Picking actions for the agenda.")
    #         actions = agenda.policy.pick_actions(agenda_state, action_history, turns_without_progress, extractions)
    #         self._log.end()
    #         self._action_history[agenda.name].extend(actions)

    #         if not done_flag:
    #             self._log.add("The agenda is not in a terminal state, so keeping it as current.")
    #             # Keep going with this agenda.
    #             self._log.end()
    #             return actions
    #         else:
    #             self._log.add("The agenda is in a terminal state, so will be stopped.")
    #             # We inactivate this agenda. Will choose a new agenda
    #             # in the main while-loop below.
    #             # We're either done with the agenda, or had too many turns
    #             # without progress.
    #             # Do last action if there is one.
    #             agenda_state.reset()
    #             self._current_agenda = None
    #             del self._active_agendas[agenda.name] #remove current agenda from the active list
    #             last_agenda = agenda
    #     self._log.end()
    # Try to pick a new agenda.
    # self._log.begin("Trying to find a new agenda to start.")
    # for agenda in np.random.permutation(self._agendas):
    #     agenda_state = agenda_states[agenda.name]
    #     self._log.begin(f"Considering agenda {agenda.name}.")

    #     if agenda == last_agenda:
    #         self._log.add("Just stopped this agenda, will not start it immediately again.")
    #         self._log.end()
    #         continue
    #     elif self._times_made_current[agenda.name] > 0:
    #         self._log.add(f"This agenda has already been used {self._times_made_current[agenda.name]} times, " +
    #                       "will not start it again.")
    #         self._log.end()
    #         continue

    #     if agenda.name in self._active_agendas: #agenda that already been kicked off
    #         # If we can kick off, make this our active agenda, do actions and return.
    #         self._log.add("The agenda can kick off. This is our new agenda!")
    #         self._log.begin(f"Puppeteer policy state for {agenda.name}:")
    #         self._log.add(f"Turns without progress: {self._turns_without_progress[agenda.name]}")
    #         self._log.add(f"Times used: {self._times_made_current[agenda.name]}")
    #         self._log.add(f"Action history: {[a.name for a in self._action_history[agenda.name]]}")
    #         self._log.end()

    #         # TODO When can the agenda be done already here?
    #         done_flag = agenda.policy.is_done(agenda_state)
    #         agenda_state.reset()

    #         # Make this our current agenda.
    #         self._current_agenda = agenda
    #         self._times_made_current[agenda.name] += 1

    #         # Do first action.
    #         # TODO run_puppeteer() uses [] for the action list, not self._action_history
    #         self._log.begin("Picking actions for the agenda.")
    #         new_actions = agenda.policy.pick_actions(agenda_state, [], 0, extractions)
    #         self._log.end()
    #         actions.extend(new_actions)
    #         self._action_history[agenda.name].extend(new_actions)

    #         # TODO This is the done_flag from kickoff. Should check again now? Probably better to enforce in Agenda
    #         # that start states are never terminal.
    #         if done_flag:
    #             self._log.add("We started the agenda, but its start state is a terminal state, so stopping it.")
    #             self._log.add("Finishing act phase without a current agenda.")
    #             self._current_agenda = None
    #             del self._active_agendas[agenda.name] #remove current agenda from the active list
    #         self._log.end()
    #         self._log.end()
    #         return actions
    #     self._log.end()
    # self._log.end()

    # # We failed to take action with an old agenda
    # # and failed to kick off a new agenda. We have nothing.
    # self._log.add("Finishing act phase without a current agenda.")

    # return actions

class Puppeteer:
    """Agendas-based dialog bot.

    A Puppeteer instance is responsible for handling all aspects of the computer's side of the conversation, making
    decisions on conversational (or other) actions to take based on the conversational state and, possibly, other
    information about the world. There is a one-to-one relationship between Puppeteers and conversations, i.e., a
    Puppeteer handles a single conversation, and a conversation is typically handled by a single Puppeteer.

    A Puppeteer delegates most of its responsibilities to other classes, most notably:
    - Agenda: The Puppeteer's behavior is largely defined by a set of agendas associated with the Puppeteer. An
        Agenda can be described as a dialog mini-bot handling a specific topic or domain of conversation with a very
        specific and limited goal, e.g., getting to know the name of the other party. A Puppeteer's conversational
        abilities are thus defined by the collective abilities of the Agendas it can use.
    - PuppeteerPolicy: A PuppeteerPolicy is responsible for picking an agenda to use based on the conversational state,
        switching between agendas when appropriate. The choice of policy can be made through the Puppeteer's
        constructor, with the DefaultPuppeteerPolicy class as the default choice if no other class is specified.

    Architecturally, the Puppeteer is implemented much as a general agent, getting information about the world through
    observations and reacting by selecting actions appropriate for some goal. Its main purpose is to be used as a dialog
    bot, but could probably be used for other purposes as well.

    A Puppeteer session consists of first creating the Puppeteer, defining its set of agendas and its policy. Then the
    conversation is simply a series of turns, each turn having the following sequence of events:
        1. The other party acts, typically some kind of conversational action.
        2. The implementation surrounding the Puppeteer registers the actions of the other party, and possibly other
           useful information about the world.
        3. The information gathered is fed to the Puppeteer through its react() method. The Puppeteer chooses a sequence
           of actions to take based on the information.
        4. The surrounding implementation takes the Puppeteer's action, and realizes them, typically providing some kind
           of reply to the other party.
    """

    def __init__(self, agendas: List[Agenda],
                 policy_cls: Type[PuppeteerPolicy] = DefaultPuppeteerPolicy,
                 plot_state: bool = False) -> None:
        """Initialize a new Puppeteer.

        Args:
            agendas: List of agendas to be used by the Puppeteer.
            policy_cls: The policy delegate class to use.
            plot_state: If true, the updated state of the current agenda is plotted after each turn.
        """
        self._agendas = agendas
        self._last_actions: List[Action] = []
        self._policy = policy_cls(agendas)
        self._plot_state = plot_state

        if self._plot_state:
            plt.ion()
            agenda_states = {}
            for a in agendas:
                fig, ax = plt.subplots()
                agenda_states[a.name] = AgendaState(a, fig, ax)
            self._agenda_states = agenda_states
        else:
            self._agenda_states = {a.name: AgendaState(a, None, None) for a in agendas}

        self._log = Logger()

    @property
    def log(self) -> str:
        """Returns a log string from the latest call to react().

        The log string contains information that is helpful in understanding the inner workings of the puppeteer -- why
        it acts the way it does based on the inputs, and what its internal state is.
        """
        return self._log.log

    def react(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[List[Action], Extractions]:
        """"Picks zero or more appropriate actions to take, given the input and current state of the conversation.

        Note that the actions are only selected by the Puppeteer, but no actions are actually performed. It is the
        responsibility of the surrounding implementation to take concrete action, based on what is returned.

        Args:
            observations: A list of Observations made since the last turn.
            old_extractions: Extractions made during the whole conversation. This may also include extractions made by
                other modules based on the current turn.

        Returns:
            A pair consisting of:
            - A list of Action objects representing actions to take, in given order.
            - An updated Extractions object, combining the input extractions with any extractions made by the Puppeteer
              in this method call.
        """
        self._log.clear()
        self._log.begin("Inputs")
        self._log.begin("Observations")
        for o in observations:
            self._log.add(str(o))
        self._log.end()
        self._log.begin("Extractions")
        for name in old_extractions.names:
            self._log.add(f"{name}: '{old_extractions.extraction(name)}'")
        self._log.end()
        self._log.end()
        new_extractions = Extractions()
        active_agendas = self._policy._active_agendas
        kicked_off_agendas = self._policy._kicked_off_agendas
        self._log.begin("Update phase")
        for agenda_state in self._agenda_states.values():
            extractions = agenda_state.update(self._last_actions, observations, old_extractions, active_agendas, kicked_off_agendas)
            new_extractions.update(extractions)

        if new_extractions.has_extraction("kickoff"):
            # When one trigger from one agenda kickoff other agenda.
            kickoff_agenda = new_extractions.extraction("kickoff")["kickoff_agenda"]
            kickoff_trigger = new_extractions.extraction("kickoff")["kickoff_trigger"]
            #print("kickoff agenda: {}, kickoff: trigger: {}".format(kickoff_agenda, kickoff_trigger))
            kickoff_agenda_state = self._agenda_states[kickoff_agenda]
            intent_observation = IntentObservation()
            intent_observation.add_intent(kickoff_trigger)
            extractions = kickoff_agenda_state.update(self._last_actions, [intent_observation], old_extractions, active_agendas, kicked_off_agendas)
            new_extractions.update(extractions)

        for agenda in self._agendas:
            # Update the list of active agendas (if any agendas are kicked off)
            agenda_state = self._agenda_states[agenda.name]
            if agenda.policy.can_kick_off(agenda_state):
                active_agendas[agenda.name] = agenda

        if self._plot_state:
            # If plot_state is enabled
            for agenda_state in self._agenda_states.values():
                agenda_state.plot()

        self._log.end()
        self._log.begin("Act phase")
        self._last_actions = self._policy.act(self._agenda_states, new_extractions)
        self._log.end()
        self._log.begin("Outputs")
        self._log.begin("Actions")
        for a in self._last_actions:
            self._log.add(str(a))
        self._log.end()
        self._log.begin("Extractions")
        for name in new_extractions.names:
            self._log.add(f"{name}: '{new_extractions.extraction(name)}'")
        self._log.end()
        self._log.end()
        
        return self._last_actions, new_extractions

    def get_active_agenda_names(self) -> List[str]:
        """ Returns the active agenda names (the ones already kicked off).
        """
        return list(self._policy._active_agendas.keys())

    def get_active_states(self, active_agenda_names) -> Dict[str, tuple[str, float, str]]:
        """ Given the active agenda names, returns the most likely current state with it probability 
            and turns_without_progress (number of consecutive turns the agenda has been idle) for each agenda name.
        """
        active_states = {}
        for agenda_name in active_agenda_names:
            current_state_probability_map = self._agenda_states[agenda_name]._state_probabilities._probabilities
            current_state_name = max(current_state_probability_map, key=lambda x: current_state_probability_map[x])
            turns_without_progress = self._policy._turns_without_progress[agenda_name]
            turns_without_progress = str(turns_without_progress) if turns_without_progress != -1 else "NA"
            active_states[agenda_name] = (current_state_name, current_state_probability_map[current_state_name], turns_without_progress)

        return active_states

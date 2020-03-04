from typing import List

import numpy as np

from agenda import Agenda, Puppeteer
from observation import MessageObservation
from trigger_detectors.loader import MyTriggerDetectorLoader

# import get_location
# import make_payment


class TestConversation:
    def __init__(self, agendas: List[Agenda]):
        
        self._puppeteer = Puppeteer(agendas)
        self._extractions = {"first_name": "Mr", "last_name": "X"}
        np.random.seed(0)

    
    def say(self, text):

        print("-"*40)
        print("You said: %s" % text)

        msg = MessageObservation(text)
        msg.add_intent("payment")
        (actions, extractions) = self._puppeteer.react([msg], self._extractions)

        print("-"*40)

        if extractions:
            print("Extractions:")
            for (key, value) in extractions.items():
                print("    %s: %s" % (key, value))
        else:
            print("No extractions")

        if self._puppeteer._policy_state._current_agenda is None:
            print("No current agenda")
        else:
            print("Current agenda: %s" % self._puppeteer._policy_state._current_agenda.name)

        print("Agenda state probabilities")
        for (agenda_name, belief) in self._puppeteer._beliefs.items():
            # TODO Hacky access to state probabilities.
            tpm = belief._transition_probability_map
            print("    %s:" % agenda_name)
            for (state_name, p) in tpm.items():
                print("        %s: %.3f" % (state_name, p))
        
        if actions:
            print("Actions:")
            for a in actions:
                print("    %s" % a)
        else:
            print("No actions")

        return (actions, extractions)


if __name__ == "__main__":
    #agendas = [get_location.create_agenda(), make_payment.create_agenda()]
    #agendas[0].store("agendas/get_location.yaml")
    #agendas[1].store("agendas/make_payment.yaml")
    # Set up trigger detector loader.
    trigger_detector_loader = MyTriggerDetectorLoader(default_snips_path="../turducken/data/training/puppeteer")
    
    # Load agendas
    get_location = Agenda.load("agendas/get_location.yaml", trigger_detector_loader)
    make_payment = Agenda.load("agendas/make_payment.yaml", trigger_detector_loader)
    agendas = [get_location, make_payment]
 
    tc = TestConversation(agendas)
    tc.say("Hello")
    tc.say("Why?")
    tc.say("routing number: 8998 account number: 12321312321")
    tc.say("None of your business")
    tc.say("No way")
    tc.say("routing number: 8998 account number: 12321312321")
    tc.say("No way")
    #tc.say("I live in Chicago")



# "Hello"
# "None of your business"
# "No way"
# "I live in Chicago"

# "routing number: 8998 account number: 12321312321"
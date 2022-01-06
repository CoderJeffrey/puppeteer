from typing import List
import numpy as np

from puppeteer import Agenda, Extractions, MessageObservation, Puppeteer
from puppeteer.trigger_detectors.loader import MyTriggerDetectorLoader


class TestConversation:
    def __init__(self, agendas: List[Agenda]):
        self._puppeteer = Puppeteer(agendas, plot_state=True)
        self._extractions = Extractions()
        np.random.seed(0)

    def say(self, text):
        print("-"*40)
        print("You said: %s" % text)

        msg = MessageObservation(text)
        msg.add_intent("payment")
        (actions, extractions) = self._puppeteer.react([msg], self._extractions)
        print(self._puppeteer.log)

        return (actions, extractions)


if __name__ == "__main__":
    # Set up trigger detector loader
    trigger_detector_loader = MyTriggerDetectorLoader(default_snips_path="training_data")

    # Load agendas
    get_location = Agenda.load("puppeteer/agendas/get_location.yaml", trigger_detector_loader)
    make_payment = Agenda.load("puppeteer/agendas/make_payment.yaml", trigger_detector_loader)
    agendas = [get_location, make_payment]
    #agendas = [make_payment]
    #agendas = [get_location]
    
    tc = TestConversation(agendas)
    results = []

    print('Type \'exit\' to terminate')
    while True:
        txt = input('send: ')
        if txt == 'exit':
            break
        results.append((tc.say(txt)))
    print(results)


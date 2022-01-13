from typing import List
import numpy as np

from puppeteer import Agenda, Extractions, MessageObservation, Puppeteer
from puppeteer.trigger_detectors.loader import MyTriggerDetectorLoader


class TestConversation:
    def __init__(self, agendas: List[Agenda]):
        self._puppeteer = Puppeteer(agendas, plot_state=True)
        self._extractions = Extractions()
        np.random.seed(0)

    def say(self, text, intent=None):
        print("-"*40)
        print("You said: %s" % text)

        msg = MessageObservation(text)
        #msg.add_intent("payment")
        if intent != None:
            msg.add_intent(intent)
        (actions, extractions) = self._puppeteer.react([msg], self._extractions)
        print(self._puppeteer.log)

        return (actions, extractions)


if __name__ == "__main__":
    # Set up trigger detector loader
    trigger_detector_loader = MyTriggerDetectorLoader("training_data")

    # Load agendas
    #get_location = Agenda.load("puppeteer/agendas/get_location.yaml", trigger_detector_loader)
    #make_payment = Agenda.load("puppeteer/agendas/make_payment.yaml", trigger_detector_loader)
    get_website = Agenda.load("puppeteer/agendas/get_website.yml", trigger_detector_loader)
    print(str(get_website))
    #agendas = [get_location, make_payment]
    #agendas = [make_payment]
    #agendas = [get_location]
    agendas = [get_website]


    tc = TestConversation(agendas)
    results = []

    print('Type \'exit\' to terminate')
    while True:
        txt = input('text: ')
        if txt == 'exit':
            break
        
        intent = input('intent (type \'NA\' to skip): ')
        if intent != 'NA':
            results.append((tc.say(txt, intent)))
        else:
            results.append((tc.say(txt)))
        '''
        results.append((tc.say(txt)))
        '''

    print(results)


import argparse
import os
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

def print_args(args):
    print("agenda_path: {}".format(args.p))
    print("agendas: {}".format(str(args.a)))
    print("training_data_path: {}".format(args.t))

def demo(args):
    print_args(args)
    agenda_path = args.p
    agenda_names = args.a
    training_data_path = args.t
    
    # Set up trigger detector loader
    trigger_detector_loader = MyTriggerDetectorLoader(training_data_path, agenda_names)

    # Load agendas
    agendas = []
    for a in agenda_names:
        yml = "{}.yaml".format(a)
        path = os.path.join(agenda_path, yml)
        agenda = Agenda.load(path, trigger_detector_loader)
        print(str(agenda))
        agendas.append(agenda)
    
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

        #results.append((tc.say(txt)))

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Puppeteer demo",
        fromfile_prefix_chars='@')
    parser.add_argument('-p', type=str, help='path to agenda directory')
    parser.add_argument('-a', nargs='+', type=str, help='list of agenda names')
    parser.add_argument('-t', type=str, help='path to training data directory')
    args = parser.parse_args()

    demo(args)

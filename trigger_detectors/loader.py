from os.path import dirname, join, realpath

from puppeteer import SpacyEngine, TriggerDetectorLoader
# from .location import CityInExtractionsTriggerDetector, LocationInMessageTriggerDetector
from .website import WebsiteTriggerDetector, WebsiteUrlTriggerDetector
from .shipment import ShipmentTriggerDetector, ShipmentNliTriggerDetector
from .payment import PaymentTriggerDetector, EAccountTriggerDetector, AskPaymentTriggerDetector, SignupTriggerDetector


class MyTriggerDetectorLoader(TriggerDetectorLoader):
    
    def __init__(self, default_snips_path=None, agenda_names=None):
        super(MyTriggerDetectorLoader, self).__init__(default_snips_path=default_snips_path)
       
        agenda_names = set(agenda_names) 
        # Our custom trigger detectors.
        
        # Used by get_website
        if "get_website" in agenda_names:
            self.register_detector(WebsiteTriggerDetector("website")) #kickoff
            self.register_detector_for_agenda("get_website", WebsiteUrlTriggerDetector("url"))

        # Used by get_shipment_nli
        if "get_shipment" in agenda_names:
            self.register_detector(ShipmentTriggerDetector("shipment")) #kickoff
            self.register_detector_for_agenda("get_shipment", ShipmentNliTriggerDetector("nli"))

        # Used by get_payment
        if "get_payment" in agenda_names:
            self.register_detector(PaymentTriggerDetector("payment")) #kickoff
            self.register_detector_for_agenda("get_payment", EAccountTriggerDetector("e_account"))
            self.register_detector_for_agenda("get_payment", AskPaymentTriggerDetector("ask_payment_response"))
            self.register_detector_for_agenda("get_payment", SignupTriggerDetector("signup"))

        '''
        # Used by get_location
        nlp = SpacyEngine.load()
        rootdir = dirname(realpath(__file__))
        snips_paths = [join(rootdir, "../../training_data/get_location/i_live")]
        cities_path = join(rootdir, "../../dictionaries/cities.txt")
        self.register_detector(CityInExtractionsTriggerDetector())
        self.register_detector(LocationInMessageTriggerDetector(snips_paths, cities_path, nlp))
        '''

        '''
        # Used by get_location
        nlp = SpacyEngine.load()
        rootdir = dirname(realpath(__file__))
        #snips_paths = [join(rootdir, "../../turducken/data/training/puppeteer/get_location/i_live")]
        #cities_path = join(rootdir, "../../turducken/data/dictionaries/cities.txt")
        snips_paths = [join(rootdir, "../../training_data/get_location/i_live")]
        #print('MTGL: snips_paths={}'.format(snips_paths))
        cities_path = join(rootdir, "../../dictionaries/cities.txt")
        #print('MTGL: cities_paths={}'.format(cities_path))
        self.register_detector(CityInExtractionsTriggerDetector())
        self.register_detector(LocationInMessageTriggerDetector(snips_paths, cities_path, nlp))
        '''


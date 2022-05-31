import requests
from typing import List, Mapping, Tuple, Union
from ..observation import Observation, MessageObservation
from ..extractions import Extractions
from ..trigger_detector import TriggerDetector
from urlextract import URLExtract

extractor = URLExtract()

def extract_url(msg: str) -> Union[Extractions, None]:
    extract_urls = extractor.find_urls(msg)
    if extract_urls:
        print('extracted urls: {}'.format(str(extract_urls)))
        valid_url = []
        invalid_url = []
        for i, url in enumerate(extract_urls):
            if "http" not in url:
                url = "http://" + url
            try:
                request_response = requests.head(url)
                if request_response.status_code == 404:
                    print("{}) {} is valid but not reachable.".format(i, url))
                    invalid_url.append(url)
                else:
                    print("{}) {} is valid and reachable.".format(i, url))
                    valid_url.append(url)
            except:
                print("{}) is invalid.".format(i, url))
                invalid_url.append(url)

        extractions = Extractions()
        if valid_url:
            extractions.add_extraction("valid_url", valid_url)
        if invalid_url:
            extractions.add_extraction("invalid_url", invalid_url)
        return extractions
    else:
        return None

class URLWebsiteTriggerDetector(TriggerDetector):

    def __init__(self, detector_name="url"):
        self._detector_name = detector_name

    @property
    def trigger_names(self) -> List[str]:
        return ["valid_url", "invalid_url"]

    def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
        extractions = Extractions()
        # Assume we have only one observation
        if isinstance(observations[0], MessageObservation):
            ext = extract_url(observations[0].text) #binary score: {0, 1}
            if ext:
                extractions.update(ext)

        if extractions.has_extraction("valid_url"):
            return ({"valid_url": 1.0, "invalid_url": 0.0}, extractions)
        elif extractions.has_extraction("invalid_url"):
            return ({"valid_url": 0.0, "invalid_url": 1.0}, extractions)
        else:
            return ({"valid_url": 0.0, "invalid_url": 0.0}, extractions)


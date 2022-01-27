import os
import errno
from typing import Optional

# look up path function
def lookfor(trigger_name: str, agenda_name: str, rootpath: str) -> Optional[str]:
	agenda_training_data_path = os.path.join(rootpath, agenda_name)
	if not os.path.isdir(agenda_training_data_path):
		raise FileNotFoundError(
			errno.ENOENT, os.strerror(errno.ENOENT), agenda_training_data_path)
	
	for d in os.listdir(agenda_training_data_path):
		if trigger_name == d:
			return os.path.join(agenda_training_data_path, d)

	return None

print(lookfor("why", "get_website", "training_data"))
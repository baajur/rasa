import logging
import os
from pathlib import Path
from typing import Text, Optional, Dict, List, Type, Union

import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.training.story_reader.story_reader import StoryReader
from rasa.core.training.structures import StoryStep
from rasa.data import YAML_FILE_EXTENSIONS, MARKDOWN_FILE_EXTENSION

logger = logging.getLogger(__name__)


def _get_reader(
    filename: Text,
    domain: Domain,
    interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
) -> Optional[StoryReader]:
    from rasa.core.training.story_reader import yaml_story_reader, markdown_story_reader

    if filename.endswith(MARKDOWN_FILE_EXTENSION):
        return markdown_story_reader.MarkdownStoryReader(
            interpreter, domain, template_variables, use_e2e, filename
        )
    elif filename.split(".")[-1] in YAML_FILE_EXTENSIONS:
        return yaml_story_reader.YAMLStoryReader(
            interpreter, domain, template_variables, use_e2e, filename
        )

    return None


async def load_data_from_resource(
    resource: Union[Text, Path],
    domain: Domain,
    interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> List[StoryStep]:
    """Loads core training data from the specified folder

    Args:
        resource: Folder/File with core training data files
        domain: Domain object
        interpreter: Interpreter to be used for parsing user's utterances
        template_variables: Variables that have to be replaced in the training data
        use_e2e: Identifies if the e2e model should be used
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training

    Returns:
        List of story steps from the training data
    """
    if not os.path.exists(resource):
        raise ValueError(f"Resource '{resource}' does not exist.")

    return await load_data_from_files(
        io_utils.list_files(resource),
        domain,
        interpreter,
        template_variables,
        use_e2e,
        exclusion_percentage,
    )


async def load_data_from_files(
    story_files: List[Text],
    domain: Domain,
    interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> List[StoryStep]:
    """

    Args:
        story_files: List of files with training data in it
        domain: Domain object
        interpreter: Interpreter to be used for parsing user's utterances
        template_variables: Variables that have to be replaced in the training data
        use_e2e: Identifies whether the e2e model should be used
        exclusion_percentage: Identifies the percentage of training data that
                              should be excluded from the training

    Returns:
        List of story steps from the training data
    """
    story_steps = []

    for story_file in story_files:

        reader = _get_reader(
            story_file, domain, interpreter, template_variables, use_e2e,
        )
        if not reader:
            logger.warning(
                f"Couldn't find a story reader for {story_file}, it will be skipped."
            )
            continue

        steps = await reader.read_from_file(story_file)
        story_steps.extend(steps)

        # if exclusion percentage is not 100
    if exclusion_percentage and exclusion_percentage != 100:
        import random

        idx = int(round(exclusion_percentage / 100.0 * len(story_steps)))
        random.shuffle(story_steps)
        story_steps = story_steps[:-idx]

    return story_steps

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import importlib


from src.features.helpers import filter_features, optimize_features
from src.llm.prompt_builder import (
    construct_prompt_code_generation,
    construct_prompt_name_description,
    construct_prompt_reflection,
)
from src.utils.feature import FeatureInfo, get_naive_features, list_to_str


logger = logging.getLogger(__name__)


class FeatureFinder(ABC):
    def __init__(self):
        """Initialize the instance."""
        self.logging_artifacts = None

    @abstractmethod
    def generate_features(self, X, y):
        """Generate features."""
        pass

    @abstractmethod
    def get_optimal_feature(
        self,
        X,
        y,
        min_samples_leaf,
        splitting_criterion,
        class_weight: Mapping[Any, float] | None = None,
        **kwargs,
    ):
        """Get optimal feature."""
        pass


class VanillaFeatureFinder(FeatureFinder):
    def __init__(self):
        """Initialize the instance."""
        super().__init__()

    def generate_features(self, X: pd.DataFrame, **kwargs):
        """Generate features."""
        list_features = []
        for col in X.columns:
            # Create closure to capture current col value
            def make_feature(column=col):
                """Make feature."""
                return lambda x: x[column]

            feature = FeatureInfo(fn=make_feature(), name=col, description=col)
            list_features.append(feature)
        return list_features

    def get_optimal_feature(
        self,
        X,
        y,
        min_samples_leaf,
        splitting_criterion=None,
        class_weight: Mapping[Any, float] | None = None,
        **kwargs,
    ):
        """Get optimal feature."""
        list_features = self.generate_features(X, **kwargs)
        if not list_features:
            return None

        list_features = optimize_features(
            splitting_criterion=splitting_criterion,
            list_features=list_features,
            X=X,
            y=y,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )
        list_features = sorted(list_features, key=lambda x: x.score, reverse=True)
        return list_features[0] if list_features else None


class LLMFeatureFinder(FeatureFinder):
    def __init__(
        self,
        llm_engine,
        parsing_module_name="src.utils.parsing",
        type_engine="old",
    ):
        """Initialize the instance."""
        super().__init__()
        self.llm_engine = llm_engine
        self.type_engine = type_engine
            
        # Dynamically import the chosen module
        parsing_module = importlib.import_module(parsing_module_name)
        
        # Assign the functions from the module to instance methods
        # Now, self.parse_code_feature will point to the correct version
        self.parse_code_feature = parsing_module.parse_code_feature
        self.parse_proposed_feature = parsing_module.parse_proposed_feature
            

    def get_optimal_feature(
        self,
        X,
        y,
        min_samples_leaf,
        target_name,
        history,
        dataset_info,
        context_template_name_desc,
        context_code,
        instructions_name_desc,
        instructions_code,
        include_ic_samples,
        include_history,
        include_feature_description,
        use_description_in_history=True,
        splitting_criterion=None,
        n_ic_samples=2,
        population_size=10,
        n_samples_per_prompt=10,
        n_outcomes_per_sample_code=1,
        use_reflection=False,
        list_instructions_reflection=None,
        n_reflections=1,
        remove_duplicates=False,
        initialize_with_naive_features=False,
        include_code_in_reflection=False,
        include_rationale_in_reflection=True,
        selector=None,
        n_prompts=1,
        n_parents_per_prompt=1,
        sort_reflection_by_score=True,
        reflection_banner_length=42,
        output_rationale=True,
        class_weight: Mapping[Any, float] | None = None,
        **kwargs,
    ):
        # Set the logging artifacts to none
        """Get optimal feature."""
        self.logging_artifacts = []

        prompt_name_description_generation = [
            construct_prompt_name_description(
                X=X,
                target_name=target_name,
                dataset_info=dataset_info,
                history=history,
                context_template=context_template_name_desc,
                instructions=instructions_name_desc,
                include_ic_samples=include_ic_samples,
                include_history=include_history,
                include_feature_description=include_feature_description,
                n_ic_samples=n_ic_samples,
                use_description_in_history=use_description_in_history,
            )
        ]

        list_features = self.generate_features(
            X=X,
            prompts_first_step=prompt_name_description_generation,
            context_code=context_code,
            instructions_code=instructions_code,
            n_outcomes_per_sample_name_desc=n_samples_per_prompt,
            n_outcomes_per_sample_code=n_outcomes_per_sample_code,
            output_rationale=output_rationale,
        )

        if initialize_with_naive_features:
            logger.info("Incorportaing the naive features in the initial population")
            naive_features = get_naive_features(X=X)

            list_features.extend(naive_features)

        list_features = optimize_features(
            splitting_criterion=splitting_criterion,
            list_features=list_features,
            X=X,
            y=y,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )

        list_original_features = copy(list_features)

        # Do Top-K selection
        list_features = sorted(list_features, key=lambda x: x.score, reverse=True)
        list_features = list_features[:population_size]

        

        # Add this list of features to the logging artifacts
        artifact = {"Type": "initial", "Features": copy(list_features)}
        self.logging_artifacts.append(artifact)

        logger.info(
            f"List features after initialization and Top-K selection: {list_to_str(list_features)}"
        )

        if use_reflection:
            if not list_instructions_reflection:
                logger.warning(
                    "use_reflection=True but no reflection instructions were provided; skipping reflection."
                )
                use_reflection = False
            elif selector is None:
                logger.warning(
                    "use_reflection=True but selector is None; skipping reflection."
                )
                use_reflection = False

        if use_reflection:
            logger.info("Using reflection" + reflection_banner_length * "-")
            for k in range(n_reflections):
                logger.info(f"Reflection Iteration number {k + 1}")
                prompts_reflection = []

                for instructions_reflection in list_instructions_reflection:
                    for _ in range(n_prompts):
                        prompts_reflection.append(
                            construct_prompt_reflection(
                                X=X,
                                list_features=list_features,
                                target_name=target_name,
                                dataset_info=dataset_info,
                                history=history,
                                context_template=context_template_name_desc,
                                instructions=instructions_reflection,
                                include_ic_samples=include_ic_samples,
                                include_history=include_history,
                                include_feature_description=include_feature_description,
                                sort_by_score=sort_reflection_by_score,
                                n_ic_samples=n_ic_samples,
                                include_code=include_code_in_reflection,
                                include_rationale=include_rationale_in_reflection,
                                selector=selector,
                                n_parents_per_prompt=n_parents_per_prompt,
                            )
                        )

                list_features_reflection = self.generate_features(
                    X=X,
                    prompts_first_step=prompts_reflection,
                    context_code=context_code,
                    instructions_code=instructions_code,
                    n_outcomes_per_sample_name_desc=n_samples_per_prompt,
                    n_outcomes_per_sample_code=n_outcomes_per_sample_code,
                    output_rationale=output_rationale,
                )

                logger.info(
                    f"List features generated by reflection: {list_to_str(list_features_reflection)}"
                )

                # optimize the features

                list_features_reflection = optimize_features(
                    splitting_criterion=splitting_criterion,
                    list_features=list_features_reflection,
                    X=X,
                    y=y,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                )

              

                artifact = {
                    "Type": f"Reflection {k} generation",
                    "Features": copy(list_features_reflection),
                }
                self.logging_artifacts.append(artifact)

                # do top k selection
                list_initial_and_reflection = list_features + list_features_reflection

                if remove_duplicates:
                    # remove
                    list_initial_and_reflection = list(set(list_initial_and_reflection))

                

                list_initial_and_reflection = sorted(
                    list_initial_and_reflection, key=lambda x: x.score, reverse=True
                )

                # Top-K selection
                list_features = (
                    list_initial_and_reflection[:population_size]
                    if population_size < len(list_initial_and_reflection)
                    else list_initial_and_reflection
                )

                artifact = {
                    "Type": f"Reflection {k} selection",
                    "Features": copy(list_features),
                }
                self.logging_artifacts.append(artifact)

        # Sort the features

        if len(list_features) == 0:
            logger.error("list features is empty, returning empty list")
            list_features = list_original_features
            assert len(list_features) > 0, "List features should not be empty"
        sorted_features = sorted(list_features, key=lambda x: x.score, reverse=True)


        best_feature = sorted_features[0]

        return best_feature

    def generate_features(
        self,
        X,
        prompts_first_step,
        context_code,
        instructions_code,
        n_outcomes_per_sample_name_desc=1,
        n_outcomes_per_sample_code=1,
        output_rationale=True,
        **kwargs,
    ):
        # The prompts first step is the prompt which ask for the semantic representations.
        """Generate features."""
        if not prompts_first_step:
            logger.warning("No prompts provided for feature generation. Returning empty list.")
            return []
        logger.info(prompts_first_step[-1])

        # Generate the features
        responses_name_desc = asyncio.run(
            self.llm_engine.query_llm(
                prompts_first_step, n_outcomes_per_sample_name_desc
            )
        )

        # Parse the features
        list_features_name_desc, total_token_used = self.parse_proposed_feature(
            responses_name_desc, output_rationale=output_rationale
        )

        # logger.info(f"List features: {list_features_name_desc}")

        # Flatten the list
        list_features_name_desc = [
            item for sublist in list_features_name_desc for item in sublist
        ]

        list_prompts_code = []
        # Loop over the features / Generate prompt for code generation / Generate the responses / Parse the responses
        for proposed_feature in list_features_name_desc:
            prompt_code_generation = construct_prompt_code_generation(
                X=X,
                context=context_code,
                proposed_feature=proposed_feature,
                instructions=instructions_code,
            )
            list_prompts_code.append(prompt_code_generation)
        if not list_prompts_code:
            logger.warning("No code-generation prompts could be constructed. Returning empty list.")
            return []
        logger.info(list_prompts_code[-1])
        responses_code = asyncio.run(
            self.llm_engine.query_llm(list_prompts_code, n_outcomes_per_sample_code)
        )
        list_features, total_token_used = self.parse_code_feature(
            responses_code, list_features_name_desc
        )

        # Flatten the list and return the

        list_features = [item for sublist in list_features for item in sublist]

        n_features_before_filter = len(list_features)
        # Filter the features
        list_features = filter_features(list_features, X)
        n_features_after_filter = len(list_features)

        logger.info(
            f"Number of features before filtering: {n_features_before_filter} and after filtering: {n_features_after_filter}"
        )
        return list_features

    

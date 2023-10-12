"""Describes model data item that can have multiple categories"""
from __future__ import annotations

from functools import total_ordering
from typing import Dict, Generic, Hashable, TypeVar

from pandas import DataFrame, Series

KT = TypeVar('KT', bound=Hashable)

@total_ordering
class DataItem(Generic[KT]):
    """Describes model data item that can have multiple categories"""
    total: float
    _categories: Dict[KT, float]
    
    def __init__(self, 
                 total: float|None = None,
                 category_proportions: Dict[KT, float]|None = None,
                 category_totals: Dict[KT, float]|None = None,
                 uncategorized_total: float|None = None):
        """Initializes the data item based on:
            Option A: total value and the proportion of each category or
            Option B: Total value for each category and for uncategorized

        Args:
            total (float | None, optional): Total value for the data item.
                Must be specified for option A. Defaults to None.
            category_proportions (Dict[KT, float] | None, optional): Dictionary of
                categories and proportion of the total beloning to the category. Must
                be specified for option A. Defaults to None.
            category_totals (Dict[KT, float] | None, optional): Dictionary of categories
                and the total value of each category. Must be specified for option B. 
                Defaults to None.
            uncategorized_total (float | None, optional): Value for the uncategorized
                values. Must be specified for option B. Defaults to None.

        Raises:
            ValueError: Invalid combination of parameters defined.
        """
        if total is not None:
            self.total = total
            self._categories = {} if category_proportions is None \
                                    else category_proportions
            return
        if category_totals is not None:
            uncategorized = 0 if uncategorized_total is None else uncategorized_total
            self.total = uncategorized + sum(category_totals.values())
            self._categories = dict(zip(category_totals.keys(),
                                        [x/self.total 
                                         for x in category_totals.values()]))
            return
        raise ValueError(
            'DataItem requires either total and category_totals to be defined')
    
    def is_valid(self, epsilon: float = 0.0001) -> bool:
        """Check if the given parameters are valid. Check that the each category
            proportion is positive and the sum of proportions is below 1.0.

        Args:
            epsilon (float, optional): Number tolerance used for checking. 
                Defaults to 0.0001.

        Returns:
            bool: True if the data item is valid.
        """
        return (min(self._categories.values()) >= 0.0 - epsilon) and \
               (sum(self._categories.values()) <= 1.0 + epsilon)
    
    @property
    def uncategorized_proportion(self) -> float:
        """Returns the proportion of the value not belonging to any category.
        i.e. (1 - sum(category_proportions))

        Returns:
            float: proportion of the total 
        """
        return 1.0 - sum(self._categories.values())
    
    @property
    def uncategorized(self) -> float:
        """Returns the value of non-categoriezed portion.

        Returns:
            float: Value for non-categorized portion.
        """
        return self.total * self.uncategorized_proportion
    
    def __getitem__(self, category: KT) -> float:
        """Returns the value for single category

        Args:
            category (KT): Category name

        Returns:
            float: Value of the given category
        """
        return self.total * self._categories[category]

    def set_category_proportions(self, category_proportions: Dict[KT, float]):
        """Sets the proportion of each category.

        Args:
            category_proportions (Dict[KT, float]): Dictionary of categories
            and the proportion of total beloning to the category.
        """
        self._categories = category_proportions.copy()
    
    def update_category_proportions(self, category_proportions: Dict[KT, float]):
        """Updates subset of category proportions.

        Args:
            category_proportions (Dict[KT, float]): Dictionary of categories
            and the proportion of total beloning to the category.
        """
        self._categories.update(category_proportions)
    
    def get_categorized(self, uncategorized_as: KT|None = None) -> Dict[KT, float]:
        """Returns total values beloning to each specified category in a dictionary.

        Args:
            uncategorized_as (KT | None, optional): Creates a new key for the result
                dictionary containing the total not beloning to any other category. 
                Defaults to None.

        Raises:
            KeyError: When the given uncategorized_as key already exists in the 
                categories.

        Returns:
            Dict[KT, float]: Dictionary of total value in each category.
        """
        if uncategorized_as in self._categories:
            raise KeyError('Specified uncategorized_as key already exists in data')
        result = dict([(key, self.total*val) for key, val in self._categories.items()])
        if uncategorized_as is not None:
            result[uncategorized_as] = self.total - sum(result.values())
        return result
    
    def __add__(self, other: DataItem[KT]|float):
        """Adds two data items together

        Args:
            other (DataItem[KT]): Righthand side operand for the addition.

        Returns:
            _type_: Sum of the two dataitems
        """
        left = self.get_categorized()
        if isinstance(other, DataItem):
            right = other.get_categorized()
            
            keys = set(list(left.keys()) + list(right.keys()))
            categories = dict([(k, left.get(k, 0.0) + right.get(k, 0.0)) for k in keys])
            uncategorized = (self.total + other.total) - \
                (sum(left.values()) + sum(right.values()))
            return DataItem(category_totals=categories,
                            uncategorized_total=uncategorized)
        return DataItem(self.total+other, self._categories)
    
    def __mul__(self, other: float) -> DataItem[KT]:
        """Scales the DataItem with a float value.

        Args:
            other (float): Righthand side scalar operand

        Returns:
            DataItem[KT]: A new scaled DataItem
        """
        return DataItem(total=self.total*other,
                        category_proportions=self._categories)
    
    def __truediv__(self, other: float) -> DataItem[KT]:
        """Divides the DataItem with a float value.

        Args:
            other (float): Righthand side division operand

        Returns:
            DataItem[KT]: A new scaled DataItem
        """
        return self * (1.0 / other)
    
    def __lt__(self, other: DataItem[KT]) -> bool:
        """Less-than operator for comparing two DataItems. Comparison is
            done only on the total value.

        Args:
            other (DataItem[KT]): Righthand side operand

        Returns:
            bool: Returns true if the total of left item is less than 
                the total of the right item.
        """
        return self.total < other.total
    
    def __eq__(self, other: DataItem[KT]) -> bool:
        """Equals operator for comparing two DataItems. Comparison is
            done only on the total value.

        Args:
            other (DataItem[KT]): Righthand side operand

        Returns:
            bool: Returns true if the total of left item equals 
                the total of the right item.
        """
        return self.total == other.total

    def __repr__(self) -> str:
        """Returns the string representation of the DataItem

        Returns:
            str: String representation of the DataItem
        """
        return str(self.total) + ':' + str(self._categories)
    
    def __str__(self) -> str:
        """Returns a human readable string representation of the DataItem

        Returns:
            str: String representation of the DataItem
        """
        categories = ', '.join([
            f'{k}:{v:g}({self._categories[k]*100:.0f}%)'
            for k, v in self.get_categorized().items()] +\
                [f'?:{self.uncategorized:g}({self.uncategorized_proportion*100:.0f}%)'])
                               
        return f'{self.total:g}:[{categories}]'
        
def dataframe_to_data_items(df: DataFrame,
                            total_column: str,
                            category_columns: Dict[KT, str]) \
                                -> Series[DataItem[KT]]:
    """Builds Pandas series of DataItems from a dataframe

    Args:
        df (DataFrame): DataFrame to read in
        total (str | None): Column name for total value of the DataItem
        category_columns (Dict[KT, str] | None): Dictionary of categories and the 
            corresponding dataframe column name

    Returns:
        Series[DataItem[KT]]: Pandas Series of DataItems
    """
    def _row_to_data_item(row: Dict[str, float]):
        return DataItem(total=row[total_column],
                        category_proportions=dict(zip(category_columns.keys(),
                                        [row[x] for x in category_columns.values()])))
    return df.apply(_row_to_data_item, axis=1)

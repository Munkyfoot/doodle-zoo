"""Database module for the Doodle Zoo app."""

import os
import time
from enum import Enum
from typing import List

import tqdm
from PIL.Image import Image
from sqlalchemy import Column, Integer, String, create_engine, desc, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class DataSource(Enum):
    """Data source enum. Used to determine where the doodle came from."""

    BASE = "base"
    USER = "user"


Base = declarative_base()


class Doodle(Base):
    """Doodle database model."""

    __tablename__ = "doodles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String, nullable=False)
    path = Column(String, nullable=False, unique=True)


class Database:
    """The database for the Doodle Zoo app."""

    def __init__(
        self,
        db_name: str = "doodles",
        data_dir: str = "data",
        auto_populate: bool = True,
    ):
        """Initialize the database.

        Args:
            db_name (str, optional): The name of the database. Defaults to "doodles".
            data_dir (str, optional): The path to the data directory. Defaults to "data".
            auto_populate (bool, optional): Whether to automatically populate the database. Defaults to True.
        """

        # Create/connect the database
        self._engine = create_engine(f"sqlite:///{data_dir}/{db_name}.db")
        Base.metadata.create_all(self._engine)

        # Create the session
        self.Session = sessionmaker(bind=self._engine)

        # Set the data directories
        self._data_dir = data_dir
        self._base_data_dir = os.path.join(data_dir, "base")
        self._user_data_dir = os.path.join(data_dir, "user")

        # Populate the database if auto_populate is True
        if auto_populate:
            self.populate()

    def populate(self):
        """Populate the database with the doodles from the base data directory."""
        session = self.Session()

        # Check if the database is empty, if so, populate it
        if not session.query(Doodle).count():
            print("Populating database...")
            labels = self.get_labels(from_base_data=True)
            for label in tqdm.tqdm(labels):
                label_path = os.path.join(self._base_data_dir, label)
                for doodle in os.listdir(label_path):
                    doodle_path = os.path.join(label_path, doodle)
                    session.add(Doodle(label=label, path=doodle_path))
            session.commit()
        session.close()

    def get_labels(self, from_base_data: bool = False) -> List[str]:
        """Get the labels from the database or the base data directory.

        Args:
            from_base_data (bool, optional): Whether to get the labels from the base data directory. Defaults to False.

        Returns:
            List[str]: The list of labels."""

        # If from_base_data is True, get the labels from the base data directory
        if from_base_data:
            return os.listdir(self._base_data_dir)

        # Otherwise, get the labels from the database
        session = self.Session()
        labels = [x[0] for x in session.query(Doodle.label).distinct()]
        session.close()
        return labels

    def get_count(self, label: str) -> int:
        """Get the number of doodles in the database for the given label.

        Args:
            label (str): The label to get the count for.

        Returns:
            int: The number of doodles for the given label."""

        session = self.Session()
        count = session.query(Doodle).filter_by(label=label).count()
        session.close()
        return count

    def get_all_counts(self) -> dict:
        """Get the number of doodles for each label in the database.

        Returns:
            dict: A dictionary of the labels and their counts."""

        labels = self.get_labels()
        return {label: self.get_count(label) for label in labels}

    def get_paths(
        self, label: str, limit: int | None = None, randomize: bool = False
    ) -> List[str]:
        """Get the paths for the doodles in the database for the given label.

        Args:
            label (str): The label to get the paths for.
            limit (int, optional): The maximum number of paths to get. Returns all if set to None. Defaults to None.
            randomize (bool, optional): Whether to randomize the order of the paths. Defaults to False.

        Returns:
            List[str]: The list of paths."""

        # If the label is "" or None, return an empty list
        if not label:
            return []

        session = self.Session()
        query = session.query(Doodle.path).filter_by(label=label)
        if randomize:
            query = query.order_by(func.random())
        else:
            query = query.order_by(desc(Doodle.id))

        if limit:
            query = query.limit(limit)
        paths = [x[0] for x in query.all()]
        session.close()
        return paths

    def get_all_paths(self, limit: int | None = None, randomize: bool = False) -> dict:
        """Get the paths for doodles in the database for each label.

        Args:
            limit (int, optional): The maximum number of paths to get. Returns all if set to None. Defaults to None.
            randomize (bool, optional): Whether to randomize the order of the paths. Defaults to False.

        Returns:
            dict: A dictionary of the labels and their paths.
        """

        labels = self.get_labels()
        return {label: self.get_paths(label, limit, randomize) for label in labels}

    def add_doodle_by_path(self, label: str, path: str):
        """Add a doodle to the database by its path.

        Args:
            label (str): The label for the doodle.
            path (str): The path to the doodle."""

        session = self.Session()
        session.add(Doodle(label=label, path=path))
        session.commit()
        session.close()

    def add_doodle(
        self, label: str, doodle: Image, data_source: DataSource = DataSource.USER
    ):
        """Save a doodle and add it to the database.

        Args:
            label (str): The label for the doodle.
            doodle (Image): The doodle as a PIL Image.
            data_source (DataSource, optional): The data source for the doodle. Defaults to DataSource.USER.
        """

        # Get the save path for the doodle from the data source and label
        source_dir = (
            self._base_data_dir
            if data_source == DataSource.BASE
            else self._user_data_dir
        )
        save_dir = os.path.join(source_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{label}_{int(time.time())}.png")

        # Save the doodle
        doodle.save(save_path)

        # Add the doodle to the database
        self.add_doodle_by_path(label, save_path)

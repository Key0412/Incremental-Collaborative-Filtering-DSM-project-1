import csv
from myICF.main import myICF

def stream(filepath, delimiter='\t', max_cases=500):
    """
    This generator utility is used to simulate a data stream from tabular data.
    Args:
        filepath: path to the csv file that contains cases. Columns named 'user', 'item', and 'rating' are expected.
        delimiter: type of delimiter used to separate columns in the csv file.
        max_cases: maximum number of cases that will be yielded.
    
    """
    with open(filepath, 'r') as csvf:
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf, delimiter=delimiter)
        n=0
        for row in csvReader:
            if n == max_cases:
                break
            n+=1
            yield row['user'], row['item'], float(row['rating'])
            
class myICF_helper(myICF):
    """
    This class inherits all methods from myICF, the class used to implement Incremental Collaborative Filtering algorithm by Papagelis et al. (2005).
    Given that movielens 100k is the dataset used in this project, this class also has utility functions to recover the title of movies, in order to present the recommendations as movie titles instead of movie IDs.
    If another dataset other than movielens is used, the parameter 'description_column' must be modified to accept a proper item description column instead of the 'title' column.
    """
    def __init__(self, filepath, delimiter='\t', description_column='title', corr_threshold=0.65, high_rating=4):
        """
        This is used during the instanciation of new objects from the class myICF.
        Args:
            filepath: path to the csv file that contains cases. Columns named 'user', 'item', and 'rating' are expected.
            delimiter: type of delimiter used to separate columns in the csv file.
            description_column: column in the tabular data that contains descriptive information about items. i.e. for movielens, this column is represented by 'title', which contains the title of movies.
            corr_threshold: defines the Pearson correlation threshold for users to be considered similar. It is used during recommendation.
            high_rating: defines a rating threshold for items that are considered for recommendation. It is used during recommendation.
        """
        super().__init__()
        self.filepath = filepath
        self.delimiter = delimiter
        self.title_dict = {}
        self.descriptions_column = description_column
        
    def get_titles(self):
        """
        Recover and stores descriptions of items from the tabular data.
        The description column must be set accordingly.
        """
        # while using another dataset, change 'title' for the proper item description column
        if self.title_dict:
            return self.title_dict
        else:
            with open(self.filepath, 'r') as csvf:
                #load csv file data using csv library's dictionary reader
                csvReader = csv.DictReader(csvf, delimiter=self.delimiter)
                for row in csvReader:
                    self.title_dict[row['item']] = row[self.descriptions_column]
            return self.title_dict

    def user_favorites(self, user, n_items=10):
        """
        This utility function is used to recover the descriptions of the 'n_items' highly rated items by 'user'.
        Args:
            user: ID of the user.
            n_items: number of highly rated items for which descriptions are returned.
        """
        if not self.title_dict.items():
            self.get_titles()
        user_favorites = sorted(self.user_ratings[user].items(), key=lambda x: -x[1])[:n_items]
        return [self.title_dict[item[0]] for item in user_favorites]

    def show_recommended_titles(self, user, n_items=10):
        """
        This utility function is used to recover the descriptions of the 'n_items' recommendations made for 'user'.
        Args:
            user: ID of the user.
            n_items: number of recommendations to be made, for which descriptions are returned.
        """
        if not self.title_dict.items():
            self.get_titles()
        return [self.title_dict[item] for item in self.recommend(user=user, n_recs=n_items)]    
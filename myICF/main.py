class myICF():
    """
    This is the class used to implement Incremental Collaborative Filtering algorithm by Papagelis et al. (2005) in Incremental Collaborative Filtering for Highly-Scalable Recommendation Algorithms.
    """
    
    def __init__(self, corr_threshold=0.65, high_rating=4):
        """
        This is used during the instancing of new objects from the class myICF.
        Args:
            corr_threshold: defines the Pearson correlation threshold for users to be considered similar. It is used during recommendation.
            high_rating: defines a rating threshold for items that are considered for recommendation. It is used during recommendation.
        """
        self.user_ratings = {} # Dict to store ratings
        self.user_meta = {} # Dict to cache users info on number of ratings and average ratings
        self.user_pair_meta = {} # Dict to cache info on factors calculated for pairs of users (B, C, D, sum of ratings to co-rated items)
        self.corr_threshold=corr_threshold
        self.high_rating=high_rating
   
    def _new_user(self, user, item, rating):
        # initialize new user
        self.user_ratings[user] = {} # initializes user in ratings dict
        self.user_meta[user] = {'q': 0, 'avg.rating': 0} # initializes user in meta dict, assign number of items user has rated and avg rating of user

        # initializes pairs of existing users with new user in user_pair_meta dict
        for u in self.user_meta.keys(): 
            if u == user:
                continue
            self.user_pair_meta[(u, user)] = {'B': 0, 'C': 0, 'D': 0}
            self.user_pair_meta[(u, user)]['sum.co_ratings'] = {u: 0, user: 0, 'n': 0}
            
    def _new_rating(self, user, item, rating):
        # Submission of a new rating
        q = self.user_meta[user]['q'] # gets number of items user has rated
        A_avg_rating = self.user_meta[user]['avg.rating']
        new_avg = ( rating/( q+1 ) ) + ( q/( q+1 ) )*A_avg_rating # calculates new avg rating for active user
        delta_avg = new_avg - A_avg_rating # difference of user's previous and current avg rating
        
        for userB in self.user_meta.keys():
            if userB == user:
                continue
                        
            B_avg_rating = self.user_meta[userB]['avg.rating']
            
            if item in self.user_ratings[userB].keys():
                # User B has rated the item                
                A_sum_coratings, B_sum_coratings, n_coratings, key = self._update_get_coratings(user, userB, item, rating, new_rating=True)
                B_rating = self.user_ratings[userB][item]
                
                e = ( rating-new_avg )*( B_rating-B_avg_rating ) - delta_avg*( B_sum_coratings - n_coratings*B_avg_rating )
                f = ( rating-new_avg )**2 + n_coratings*(delta_avg**2) - 2*delta_avg*( A_sum_coratings - n_coratings*A_avg_rating )
                g = ( B_rating-B_avg_rating )**2
                
            else:
                # User B had not rated the item
                A_sum_coratings, B_sum_coratings, n_coratings, key = self._get_coratings(user, userB)
                
                e = - delta_avg*( B_sum_coratings - n_coratings*B_avg_rating )
                f = n_coratings*(delta_avg**2) - 2*delta_avg*( A_sum_coratings - n_coratings*A_avg_rating )
                g = 0
            
            for factor, increment in zip(['B', 'C', 'D'], [e, f, g]):
                self.user_pair_meta[key][factor] += increment
        
        self.user_ratings[user][item] = rating # updates rating given by user to item
        self.user_meta[user]['q'] += 1 # updates number of items user has rated
        self.user_meta[user]['avg.rating'] = new_avg # updates avg rating
        
        
    def _update_rating(self, user, item, rating):
        # Update of an existing rating
        delta_rating = rating - self.user_ratings[user][item] # difference of user's previous and current rating for item
        q = self.user_meta[user]['q'] # gets number of items user has rated
        A_avg_rating = self.user_meta[user]['avg.rating']
        new_avg = delta_rating/q + A_avg_rating # calculates new avg rating for active user
        delta_avg = new_avg - A_avg_rating # difference of user's previous and current avg rating
        
        for userB in self.user_meta.keys():
            if userB == user:
                continue
                            
            B_avg_rating = self.user_meta[userB]['avg.rating']
            
            if item in self.user_ratings[userB].keys():
                # User B has rated the item                
                A_sum_coratings, B_sum_coratings, n_coratings, key = self._update_get_coratings(user, userB, item, rating, new_rating=False)
                B_rating = self.user_ratings[userB][item]
                
                e = delta_rating*( B_rating-B_avg_rating ) - delta_avg*( B_sum_coratings - n_coratings*B_avg_rating )
                # computation of f is not clear for both cases
                f = delta_rating**2 + 2*delta_rating*( rating-new_avg ) + n_coratings*delta_avg**2 - 2*delta_avg*( A_sum_coratings - n_coratings*A_avg_rating )
                g = 0
            else:
                # User B had not rated the item
                A_sum_coratings, B_sum_coratings, n_coratings, key = self._get_coratings(user, userB)
                
                e = - delta_avg*( B_sum_coratings - n_coratings*B_avg_rating )
                # computation of f is not clear for both cases
                f = n_coratings*delta_avg**2 - 2*delta_avg*( A_sum_coratings - n_coratings*A_avg_rating )
                g = 0

            for factor, increment in zip(['B', 'C', 'D'], [e, f, g]):
                self.user_pair_meta[key][factor] += increment

        self.user_ratings[user][item] = rating # updates rating given by user to item
        self.user_meta[user]['q'] += 1 # updates number of items user has rated
        self.user_meta[user]['avg.rating'] = new_avg # updates avg rating        
        
    def _get_coratings(self, userA, userB):
        if (userB, userA) in self.user_pair_meta.keys():
            key = (userB, userA)
        else:
            key = (userA, userB)
        A_sum_coratings = self.user_pair_meta[key]['sum.co_ratings'][userA]
        B_sum_coratings = self.user_pair_meta[key]['sum.co_ratings'][userB]
        n_coratings = self.user_pair_meta[key]['sum.co_ratings']['n']
        return A_sum_coratings, B_sum_coratings, n_coratings, key
        
    def _update_get_coratings(self, userA, userB, item, rating, new_rating=True):
        if (userB, userA) in self.user_pair_meta.keys():
            key = (userB, userA)
        else:
            key = (userA, userB)
        if new_rating:            
            self.user_pair_meta[key]['sum.co_ratings'][userA] += rating
            A_sum_coratings = self.user_pair_meta[key]['sum.co_ratings'][userA]
            self.user_pair_meta[key]['sum.co_ratings'][userB] += self.user_ratings[userB][item]
            B_sum_coratings = self.user_pair_meta[key]['sum.co_ratings'][userB]
            self.user_pair_meta[key]['sum.co_ratings']['n'] += 1
            n_coratings = self.user_pair_meta[key]['sum.co_ratings']['n']
        else:
            self.user_pair_meta[key]['sum.co_ratings'][userA] += (rating - self.user_ratings[userA][item])
            A_sum_coratings = self.user_pair_meta[key]['sum.co_ratings'][userA]
            B_sum_coratings = self.user_pair_meta[key]['sum.co_ratings'][userB]
            n_coratings = self.user_pair_meta[key]['sum.co_ratings']['n']
                
        return A_sum_coratings, B_sum_coratings, n_coratings, key
    
    def run(self, user, item, rating):
        """
        This function is used to run the algorithm over cases in a data stream, updating the factors from the similarity between users incrementally.
        """
        # initialize new user
        if user not in self.user_meta.keys(): 
            self._new_user(user, item, rating)
            
        # Submission of a new rating
        if item not in self.user_ratings[user].keys(): 
            self._new_rating(user, item, rating)
            
        # Update of an existing rating
        else: 
            self._update_rating(user, item, rating)
            
    def recommend(self, user, n_recs=10):
        """
        This function is used to make recommendations for an user. 
        The parameters 'corr_threshold' and 'high_rating' defined during instanciation are used here to filter similar users and recommended items.
        Args:
            user: identification of the user, as in the overall scheme of the input data stream used for learning the model.
            n_recs: number of recommendations that are to be returned.
        """
        item_count = {}
        for userB in self.user_meta.keys():
            if userB == user:
                continue
            if (userB, user) in self.user_pair_meta.keys():
                key = (userB, user)
            else:
                key = (user, userB)

            B = self.user_pair_meta[key]['B']
            C = self.user_pair_meta[key]['C']
            D = self.user_pair_meta[key]['D']
            
            try:
                pearson_corr = round(B / ( ( C**(1/2) ) * ( D**(1/2) ) ), 2)
                if abs(pearson_corr) > 1.5: # if pearson corr is greater than 1 in module, something is wrong! C or D may be too low, or else.
                    continue
                if pearson_corr >= self.corr_threshold: # sometimes, C**(1/2) and D**(1/2) are complex (C and D negative), thus they cannot be compared
                    for item in self.user_ratings[userB].keys():
                        if item in self.user_ratings[user].keys():
                            continue
                        if self.user_ratings[userB][item] >= self.high_rating:
                            item_count.setdefault(item, 0)
                            item_count[item] += 1 # self.user_ratings[userB][item]
            except:
                continue
                
        sorted_item_count = sorted(item_count.items(), key=lambda x: -x[1])
        recommended_items = [i[0] for i in sorted_item_count[:n_recs]]
        return recommended_items 
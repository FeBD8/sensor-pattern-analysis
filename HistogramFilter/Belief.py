class Belief:
    """
    Class which apply the Bayes Filter Algorithm
     Attributes
     ------------------------
     bel : float []
        probability distribution over all possible states, conditioned by the observations so far
        bel(xt):=P(xt∣e1:t)
     bel_projected : float []
        probability distribution over all the possible states at time t, given only past observations
        bel⎯projected(xt):=P(xt∣e1:t−1)
     pos : String []
        state domain
     prob_state : list(float [])
        list of probabilities to go in a room given the current room
     sensors_error_rate : list(float [])
        list of probabilities to be in a room given the sensors' measures
     Methods
     -----------------------
     bel_upgrade(self, sensor_output)
    """

    def __init__(self, bel, pos, prob_state, ser, movement_transaction, exit_transaction, prob_exit_room):
        self.bel = bel
        self.bel_projected = [0 for x in range(0, len(pos))]
        self.pos = pos
        self.prob_state = prob_state
        self.sensors_error_rate = ser
        self._movement_transaction = movement_transaction
        self._exit_transaction = exit_transaction
        self._prob_exit_room = prob_exit_room

    @property
    def bel(self):
        return self.__bel

    @bel.setter
    def bel(self, bel):
        self.__bel = bel

    @property
    def bel_projected(self):
        return self.__bel_projected

    @bel_projected.setter
    def bel_projected(self, bel_projected):
        self.__bel_projected = bel_projected

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    @property
    def prob_state(self):
        return self.__prob_state

    @prob_state.setter
    def prob_state(self, prob_state):
        self.__prob_state = prob_state

    @property
    def sensors_error_rate(self):
        return self.__sensors_error_rate

    @sensors_error_rate.setter
    def sensors_error_rate(self, sensors_error_rate):
        self.__sensors_error_rate = sensors_error_rate

    def bel_upgrade(self, transactions):
        self._bel_projected_upgrade(transactions)

        temp = []
        if self._movement_transaction in transactions:
            i = transactions.index(self._movement_transaction)
            self.bel = [x*y for x, y in zip(self.bel_projected, self.sensors_error_rate[i][self._movement_transaction])]

        else:
            for i in range(0, len(transactions)):
                temp.append([y * x for y, x in zip(self.bel_projected, self.sensors_error_rate[i][transactions[i]])])
            self.bel = [sum(x) for x in zip(*temp)]

        eta = 1/sum(self.bel)
        self.bel = [x * eta for x in self.bel]

    def _bel_projected_upgrade(self, transactions):

        def compute_bel_projected(bel,  prob_state):
            bel_projected = [[] for x in range(0, len(bel))]
            for i in range(0, len(bel)):
                bel_projected[i] = [x * bel[i] for x in prob_state[i]]
            bel_projected = [sum(x) for x in zip(*bel_projected)]
            return bel_projected

        def compute_prob_state(prob_state, math_waste, index):
            tmp = prob_state[:]
            tmp[index] = 0
            norm = math_waste/sum(tmp)
            tmp = [norm * x for x in tmp]
            prob_state = [x+y for x, y in zip(prob_state, tmp)]

            return prob_state

        if self._movement_transaction in transactions:
            self.bel_projected = compute_bel_projected(self.bel, self.prob_state)

        else:
            index = transactions.index(self._exit_transaction)
            state = [[] for x in range(0, len(transactions))]
            for i, array in enumerate(self.prob_state):
                state[i] = self.prob_state[i]
                if self.prob_state[i][index] !=0:
                    math_waste = state[i][index] - self._prob_exit_room
                    state[i][index] = self._prob_exit_room
                    state[i] = compute_prob_state(state[i], math_waste, index)

            self.bel_projected = compute_bel_projected(self.bel, state)

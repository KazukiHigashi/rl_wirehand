from sklearn.decomposition import PCA
import numpy as np


class SynergyManager:
    def __init__(self, num_axis, init_poslist, maxn_pos, pop_order="sequence", reward_type="standard", distance="euclid"):
        self.pos_list = init_poslist
        self.target_list = []

        self.num_axis = num_axis
        self.maxn_pos = maxn_pos
        self.pop_order = pop_order
        self.reward_type = reward_type
        self.error_distance = distance

        self.axis = np.array([[0.5] * 14] * self.num_axis)  # num_axis=5
        print(self.axis)

        self.pca = PCA(self.num_axis)

    def add_pos(self, pos, target_pos):
        if len(self.pos_list) >= self.maxn_pos:
            self._pop_pos()
        self.pos_list.append(pos)
        self.target_list.append(target_pos)

    def add_list(self, partial_poslist):  # リストをリストに追加. new
        self.pos_list.extend(partial_poslist)

    def set_poslist(self, poslist):
        self.pos_list = poslist

    def get_poslist(self):
        return self.pos_list

    def get_npylist(self):
        return [self.pos_list, self.target_list]

    def get_reward_type(self):
        return self.reward_type

    # Pop the biggest outlier from the pos_list every pop sequence.
    def _pop_pos(self):
        if self.pop_order == "error":
            max_error_idx = np.argmax(np.linalg.norm(
                np.array(self.pos_list) - self.calc_inverse(self.calc_transform(self.pos_list)), axis=1))
            return self.pos_list.pop(max_error_idx), self.target_list.pop(max_error_idx)
        elif self.pop_order == "sequence":
            return self.pos_list.pop(0), self.target_list.pop(0)
        else:
            raise RuntimeError("illegal pop_order argument")
        return None

    def calc_pca(self):
        self.pca.fit(self.pos_list)

    def calc_transform(self, pos):
        t_pos = self.pca.transform(pos)
        return t_pos

    def calc_inverse(self, pos):
        i_pos = self.pca.inverse_transform(pos)
        return i_pos

    def get_variance_ratio(self):
        return self.pca.explained_variance_ratio_

    def get_components(self):
        if len(self.pos_list) > 100:
            return self.pca.components_[0]
        else:
            return [0]*14

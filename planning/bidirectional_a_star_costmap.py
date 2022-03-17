import math
import matplotlib.pyplot as plt

show_animation = True


class BidirectionalAStarCostmapPlanner:

    def __init__(self, costmap, resolution=1.0, obs=None):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.obstacle = obs
        self.min_x, self.min_y = None, None
        self.max_x, self.max_y = None, None
        self.x_width, self.y_width, self.obstacle_map = None, None, None
        self.costmap = None
        self.resolution = int(resolution)
        self.calc_obstacle_map(costmap)
        self.motion = self.get_motion_model()
        
        self.dynamic_costmap = None

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        Bidirectional A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set_A, closed_set_A = dict(), dict()
        open_set_B, closed_set_B = dict(), dict()
        open_set_A[self.calc_grid_index(start_node)] = start_node
        open_set_B[self.calc_grid_index(goal_node)] = goal_node

        current_A = start_node
        current_B = goal_node
        meet_point_A, meet_point_B = None, None

        while 1:
            if len(open_set_A) == 0:
                # print("Open set A is empty..")
                break

            # if len(open_set_B) == 0:
            #     print("Open set B is empty..")
            #     break

            c_id_A = min(
                open_set_A,
                key=lambda o: self.find_total_cost(open_set_A, o, current_B))

            current_A = open_set_A[c_id_A]

            # c_id_B = min(
            #     open_set_B,
            #     key=lambda o: self.find_total_cost(open_set_B, o, current_A))

            # current_B = open_set_B[c_id_B]

            # show graph
            # if show_animation:  # pragma: no cover
            #     plt.plot(self.calc_grid_position(current_A.x, self.min_x),
            #              self.calc_grid_position(current_A.y, self.min_y),
            #              "xc")
            #     # plt.plot(self.calc_grid_position(current_B.x, self.min_x),
            #     #          self.calc_grid_position(current_B.y, self.min_y),
            #     #          "xc")
            #     # for stopping simulation with the esc key.
            #     plt.gcf().canvas.mpl_connect(
            #         'key_release_event',
            #         lambda event: [exit(0) if event.key == 'escape' else None])
            #     if len(closed_set_A.keys()) % 10 == 0:
            #         plt.pause(0.001)

            # if current_A.x == current_B.x and current_A.y == current_B.y:
            #     print("Found goal")
            #     meet_point_A = current_A
            #     meet_point_B = current_B
            #     break
            if current_A.x == goal_node.x and current_A.y == goal_node.y:
                # print("Found goal")
                goal_node.parent_index = current_A.parent_index
                goal_node.cost = current_A.cost
                break

            # Remove the item from the open set
            del open_set_A[c_id_A]
            # del open_set_B[c_id_B]

            # Add it to the closed set
            closed_set_A[c_id_A] = current_A
            # closed_set_B[c_id_B] = current_B

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):

                # c_nodes = [self.Node(current_A.x + self.motion[i][0],
                #                      current_A.y + self.motion[i][1],
                #                      current_A.cost + self.motion[i][2],
                #                      c_id_A),
                #            self.Node(current_B.x + self.motion[i][0],
                #                      current_B.y + self.motion[i][1],
                #                      current_B.cost + self.motion[i][2],
                #                      c_id_B)]
                node = self.Node(current_A.x + self.motion[i][0],
                                     current_A.y + self.motion[i][1],
                                     current_A.cost + self.motion[i][2],
                                     c_id_A)

                # n_ids = [self.calc_grid_index(c_nodes[0]),
                #          self.calc_grid_index(c_nodes[1])]

                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set_A:
                    continue

                if n_id not in open_set_A:
                    open_set_A[n_id] = node  # discovered a new node
                else:
                    if open_set_A[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set_A[n_id] = node

                # # If the node is not safe, do nothing
                # continue_ = self.check_nodes_and_sets(c_nodes, closed_set_A,
                #                                       closed_set_B, n_ids)

                # if not continue_[0]:
                #     if n_ids[0] not in open_set_A:
                #         # discovered a new node
                #         open_set_A[n_ids[0]] = c_nodes[0]
                #     else:
                #         if open_set_A[n_ids[0]].cost > c_nodes[0].cost:
                #             # This path is the best until now. record it
                #             open_set_A[n_ids[0]] = c_nodes[0]

                # if not continue_[1]:
                #     if n_ids[1] not in open_set_B:
                #         # discovered a new node
                #         open_set_B[n_ids[1]] = c_nodes[1]
                #     else:
                #         if open_set_B[n_ids[1]].cost > c_nodes[1].cost:
                #             # This path is the best until now. record it
                #             open_set_B[n_ids[1]] = c_nodes[1]

        # rx, ry = self.calc_final_bidirectional_path(
        #     meet_point_A, meet_point_B, closed_set_A, closed_set_B)
        rx, ry = self.calc_final_path(goal_node, closed_set_A)
        rx.reverse()
        ry.reverse()
        return rx, ry

    # takes two sets and two meeting nodes and return the optimal path
    def calc_final_bidirectional_path(self, n1, n2, setA, setB):
        rx_A, ry_A = self.calc_final_path(n1, setA)
        rx_B, ry_B = self.calc_final_path(n2, setB)

        rx_A.reverse()
        ry_A.reverse()

        rx = rx_A + rx_B
        ry = ry_A + ry_B

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], \
                 [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def check_nodes_and_sets(self, c_nodes, closedSet_A, closedSet_B, n_ids):
        continue_ = [False, False]
        if not self.verify_node(c_nodes[0]) or n_ids[0] in closedSet_A:
            continue_[0] = True

        if not self.verify_node(c_nodes[1]) or n_ids[1] in closedSet_B:
            continue_[1] = True

        return continue_

    @staticmethod
    def calc_heuristic(n1, n2, costmap, dynamic_costmap=None):
        w = 1.0  # weight of heuristic
        # print(n2.x, n2.y, len(costmap), len(costmap[0]))
        if n2.x >= len(costmap):
            n2.x = len(costmap) -1
        if n2.y >= len(costmap[0]):
            n2.y = len(costmap[0])-1
        
        map_cost = 1.0 * costmap[int(n2.x)][int(n2.y)]

        if dynamic_costmap is not None:
            map_cost = 1.0 * (costmap[int(n2.x)][int(n2.y)] + dynamic_costmap[int(n2.x)][int(n2.y)])
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y) + map_cost
        return d

    def find_total_cost(self, open_set, lambda_, n1):
        g_cost = open_set[lambda_].cost
        h_cost = self.calc_heuristic(n1, open_set[lambda_], self.costmap)
        f_cost = g_cost + h_cost
        return f_cost

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def is_goal_valid(self, gx, gy):
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        # print("goal: ", goal_node.x, goal_node.y)
        if self.verify_node(goal_node):
            nx = int(math.floor(goal_node.x))
            ny = int(math.floor(goal_node.y))
            if nx >= len(self.costmap):
                nx = len(self.costmap) -1
            if ny >= len(self.costmap[0]):
                ny = len(self.costmap[0])-1
            # print(self.costmap[nx][ny])
            if self.costmap[nx][ny] < 150:
                return True

        return False
    
    def update_dynamic_obs(self, obs_list):
        self.dynamic_costmap = [[0 for _ in range(self.y_width)]
                                   for _ in range(self.x_width)]
        for ob in obs_list:
            node = self.Node(self.calc_xy_index(ob[0], self.min_x),
                             self.calc_xy_index(ob[1], self.min_y), 0.0, -1)
            obx = self.calc_grid_position(node.x, self.min_x)
            oby = self.calc_grid_position(node.y, self.min_y)
            if obx < self.min_x:
                continue
            if obx > self.max_x:
                continue
            if oby < self.min_y:
                continue
            if oby > self.max_y:
                continue
            nx = int(math.floor(node.x))
            ny = int(math.floor(node.y))
            if nx >= len(self.dynamic_costmap):
                nx = len(self.dynamic_costmap) -1
            if ny >= len(self.dynamic_costmap[0]):
                ny = len(self.dynamic_costmap[0])-1
            self.dynamic_costmap[nx][ny] = 255
            
    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # # collision check
        # if self.obstacle_map[node.x][node.y]:
        #     return False
        # nx = int(math.floor(node.x))
        # ny = int(math.floor(node.y))
        # if nx >= len(self.costmap):
        #     nx = len(self.costmap) -1
        # if ny >= len(self.costmap[0]):
        #     ny = len(self.costmap[0])-1
        # print(self.costmap[nx][ny])
        # if self.costmap[nx][ny] > 150:
        #     return False

        return True

    def calc_obstacle_map(self, costmap):



        tempx = []
        tempy = []

        # print("self.x_width",self.x_width)
        # print("self.y_width",self.y_width)

        for i in range(112):
            for j in range(202):
                if costmap[i][j]<30:
                    tempx.append(j)
                    tempy.append(i)

        if show_animation:  # pragma: no cover
            plt.plot(tempx, tempy, ".k")
            # plt.plot(sx, sy, "og")
            # plt.plot(gx, gy, "ob")
            plt.grid(True)
            plt.axis("equal")

        self.min_x = 0
        self.min_y = 0
        self.max_x = costmap.shape[1]
        self.max_y = costmap.shape[0]
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        # self.x_width = costmap.shape[1]
        # self.y_width = costmap.shape[0]
        self.x_width = int(math.floor((self.max_x - self.min_x) / self.resolution))
        self.y_width = int(math.floor((self.max_y - self.min_y) / self.resolution))
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        self.costmap = [[255 for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        self.dynamic_costmap = [[0 for _ in range(self.y_width)]
                                   for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                self.costmap[ix][iy] = 255 - costmap[y][x]
                if costmap[y][x] < 30:
                    self.obstacle_map[ix][iy] = True

        # tempx = []
        # tempy = []

        # print("self.x_width",self.x_width)
        # print("self.y_width",self.y_width)

        # for i in range(self.x_width):
        #     for j in range(self.y_width):
        #         if self.costmap[i][j]<30:
        #             tempx.append(j)
        #             tempy.append(i)

        # if show_animation:  # pragma: no cover
        #     plt.plot(tempx, tempy, ".k")
        #     # plt.plot(sx, sy, "og")
        #     # plt.plot(gx, gy, "ob")
        #     plt.grid(True)
        #     plt.axis("equal")

    @staticmethod
    def get_motion_model(step=1):
        # dx, dy, cost
        motion = [[step, 0, step],
                  [0, step, step],
                  [-step, 0, step],
                  [0, -step, step],
                  [-step, -step, math.sqrt(2*step)],
                  [-step, step, math.sqrt(2*step)],
                  [step, -step, math.sqrt(2*step)],
                  [step, step, math.sqrt(2*step)]]
        return motion


def generate_mask(radius):
    radius = int(radius)
    y, x = np.ogrid[0: 2*radius+1, 0:2*radius+1]
    mask = (x - radius)**2 + (y - radius)**2 <= radius**2
    mask = np.array(mask)
    # print(mask, np.array(mask).shape)
    indices = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]:
                indices.append([i - radius, j - radius])
    # print(indices)
    return indices


def get_costmap(img, radius):
    indices = generate_mask(radius)
    costmap = img.copy()
    cols, rows = costmap.shape
    total = cols * rows
    for i in range(cols):
        for j in range(rows):
            min_dis = radius ** 2
            for index in indices:
                ix, iy = i + index[0], j + index[1]
                if (ix >= 0) and (iy >= 0) and (ix < cols) and (iy < rows):
                    if img[ix][iy] == 0:
                        dis = index[0] ** 2 + index[1] ** 2
                        if dis < min_dis:
                            min_dis = dis
            costmap[i][j] = int(255 * float(min_dis) / (radius ** 2))
        if (i%10 == 0):
            progress = int((i*j)/total*100)
            # print('[Calculation] {0:.2f} % finished'.format(progress))
    return costmap

from tkinter import *
import numpy as np

# class GUI(Canvas):
#     def __init__(self,parent,**kwargs):
#         Canvas.__init__(self,parent,**kwargs)
#         self.bind("<Configure>", self.on_resize)
#         self.height = self.winfo_reqheight()
#         self.width = self.winfo_reqwidth()
#
#     def on_resize(self,event):
#         # determine the ratio of old width/height to new width/height
#         wscale = float(event.width)/self.width
#         hscale = float(event.height)/self.height
#         self.width = event.width
#         self.height = event.height
#         # resize the canvas
#         self.config(width=self.width, height=self.height)
#         # rescale all the objects tagged with the "all" tag
#         self.scale("all",0,0,wscale,hscale)

class GUI(Canvas):
    def __init__(self, master, *args, **kwargs):
        Canvas.__init__(self, master=master, *args, **kwargs)


def draw_square_q(polygon, x, y, q, actions, dim=50):
    polygon.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                           fill='white', width=2)

    font = ('Helvetica', '30', 'bold')

    for i, a in enumerate(actions):
        if a == 0:
            polygon.create_polygon([x + dim, y, x + dim / 2., y + dim / 2., x + dim, y + dim], outline='gray',
                                   fill='red', width=2)
            polygon.create_text(x + 3 * dim / 4., y + dim / 2., font=font, text="{:.3f}".format(q[i]), anchor='center')
        elif a == 1:
            polygon.create_polygon([x, y + dim, x + dim / 2., y + dim / 2., x + dim, y + dim], outline='gray',
                                   fill='green', width=2)
            polygon.create_text(x + dim / 2., y + 3 * dim / 4., font=font, text="{:.3f}".format(q[i]), anchor='n')
        elif a == 2:
            polygon.create_polygon([x, y, x + dim / 2., y + dim / 2., x, y + dim], outline='gray',
                                   fill='yellow', width=2)
            polygon.create_text(x + dim / 4., y + dim / 2., font=font, text="{:.3f}".format(q[i]), anchor='center')
        elif a == 3:
            polygon.create_polygon([x + dim, y, x + dim / 2., y + dim / 2., x, y], outline='gray',
                                   fill='purple', width=2)
            polygon.create_text(x + dim / 2., y + dim / 4., font=font, text="{:.3f}".format(q[i]), anchor='s')


def draw_square_policy(w, x, y, pol, actions, dim=30, width=1):
    w.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                     fill='white', width=2)

    font = ('Helvetica', '30', 'bold')
    if (hasattr(pol, "size") and pol.size > 1) or isinstance(pol, list):
        d = pol
    else:
        d = [-1] * len(actions)
        idx = actions.index(pol)
        d[idx] = 1

    for j, v in enumerate(d):
        if j < len(actions):
            a = actions[j]
            if a == 0 and v > 0:
                w.create_line(x + dim / 4., y + dim / 2., x + 3*dim / 4., y + dim / 2., tags=("line",), arrow="last",
                              width=width)
                if not np.isclose(v, 1.):
                    w.create_text(x + 3*dim / 4., y + dim / 2., font=font, text="{:.1f}".format(v), anchor='w')
            elif a == 1 and v > 0:
                w.create_line(x + dim / 2., y + dim / 4., x + dim / 2., y + 3* dim / 4., tags=("line",), arrow="last",
                              width=width)
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 2., y + 3*dim / 4., font=font, text="{:.1f}".format(v), anchor='n')
            elif a == 2 and v >0:
                w.create_line(x + 3 *dim / 4., y + dim / 2., x+dim/4., y + dim/2., tags=("line",), arrow="last",
                              width=width)
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 4., y + dim / 2., font=font, text="{:.1f}".format(v), anchor='e')
            elif a == 3 and v >0:
                w.create_line(x + dim / 2., y + 3 *dim / 4., x + dim / 2., y + dim / 4., tags=("line",), arrow="last",
                              width=width)
                if not np.isclose(v, 1.):
                    w.create_text(x + dim / 2., y + dim / 4., font=font, text="{:.1f}".format(v), anchor='s')


def render_q(env, q):
    root = Tk()
    w = GUI(root)
    rows, cols = len(env.grid), max(map(len, env.grid))
    dim = 200
    w.config(width=cols * (dim + 12), height=rows * (dim + 12))
    for s in range(env.n_states):
        r, c = env.state2coord[s]
        draw_square_q(w, 10 + c * (dim + 4), 10 + r * (dim + 4), dim=dim, q=q[s],
                      actions=env.state_actions[s])
        w.pack()
    w.pack()
    root.mainloop()


def render_policy(env, d):
    root = Tk()
    w = GUI(root, width=750, heighth=750)
    rows, cols = len(env.grid), max(map(len, env.grid))
    dim = 20
    w.config(width=cols * (dim + 12), height=rows * (dim + 12))
    for s in range(env.n_states):
        r, c = env.state2coord[s]
        draw_square_policy(w, 10 + c * (dim + 4), 10 + r * (dim + 4), dim=dim, pol=d[s],
                           actions=env.state_actions[s])
        w.pack()
    w.pack()
    root.mainloop()

def render_non_det_policy(env, policy_array):
    # policy_array[i] = int or list of int representing possible actions (0 denotes impossible, include all states)
    root = Tk()
    w = GUI(root)
    rows, cols = len(env.grid), max(map(len, env.grid))
    dim = 60
    w.config(width=cols * (dim + 12), height=rows * (dim + 12))
    for s in range(env.n_states):
        r, c = env.state2coord[s]
        allowed_actions = policy_array[s]

        draw_square_policy(w, 10 + c * (dim + 4), 10 + r * (dim + 4), dim=dim, pol=allowed_actions,
                           actions=range(4), width=3)
        w.pack()
    w.pack()
    root.mainloop()

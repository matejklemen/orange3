# pylint: disable=too-many-ancestors
from enum import IntEnum
from types import SimpleNamespace as namespace

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QLineF, QPoint
from AnyQt.QtGui import QColor, QCursor
from AnyQt.QtWidgets import QToolTip

import pyqtgraph as pg

from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.projection import FreeViz
from Orange.projection.freeviz import FreeVizModel
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.visualize.utils.widget import OWAnchorProjectionWidget
from Orange.widgets.widget import Input, Output


class Result(namespace):
    projector = None  # type: FreeViz
    projection = None  # type: FreeVizModel


MAX_ITERATIONS = 1000


def run_freeviz(data: Table, projector: FreeViz, state: TaskState):
    res = Result(projector=projector, projection=None)
    step, steps = 0, MAX_ITERATIONS
    initial = res.projector.components_.T
    state.set_status("Calculating...")
    while step < steps:
        # Needs a copy because projection should not be modified inplace.
        # If it is modified inplace, the widget and the thread hold a
        # reference to the same object. When the thread is interrupted it
        # is still modifying the object, but the widget receives it
        # (the modified object) with a delay.
        res.projection = res.projector(data).copy()
        anchors = res.projector.components_.T
        res.projector.initial = anchors

        state.set_partial_result(res)
        if np.allclose(initial, anchors, rtol=1e-5, atol=1e-4):
            return res
        initial = anchors

        step += 1
        state.set_progress_value(100 * step / steps)
        if state.is_interruption_requested():
            return res


class OWFreeVizGraph(OWGraphWithAnchors):
    hide_radius = settings.Setting(0)

    @property
    def scaled_radius(self):
        return self.hide_radius / 100 + 1e-5

    def update_radius(self):
        self.update_circle()
        self.update_anchors()

    def set_view_box_range(self):
        self.view_box.setRange(QRectF(-1.05, -1.05, 2.1, 2.1))

    def closest_draggable_item(self, pos):
        points, *_ = self.master.get_anchors()
        if points is None or not len(points):
            return None
        mask = np.linalg.norm(points, axis=1) > self.scaled_radius
        xi, yi = points[mask].T
        distances = (xi - pos.x()) ** 2 + (yi - pos.y()) ** 2
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF ** 2:
            return np.flatnonzero(mask)[np.argmin(distances)]
        return None

    def update_anchors(self):
        points, labels = self.master.get_anchors()
        if points is None:
            return
        r = self.scaled_radius
        feat_importances = []
        if self.anchor_items is None:
            self.anchor_items = []
            for point, label in zip(points, labels):
                anchor = AnchorItem(line=QLineF(0, 0, *point), text=label)
                is_visible = np.linalg.norm(point) > r
                if is_visible:
                    feat_importances.append((label, np.linalg.norm(point)))
                anchor.setVisible(is_visible)
                anchor.setPen(pg.mkPen((100, 100, 100)))
                self.plot_widget.addItem(anchor)
                self.anchor_items.append(anchor)
        else:
            for anchor, point, label in zip(self.anchor_items, points, labels):
                anchor.setLine(QLineF(0, 0, *point))
                anchor.setText(label)
                is_visible = np.linalg.norm(point) > r
                if is_visible:
                    feat_importances.append((label, np.linalg.norm(point)))
                anchor.setVisible(is_visible)
        self.feat_importances = feat_importances

        if r < (0 + 1e-3):
            print("**Feature importances**")
            for i, (lbl, importance) in enumerate(sorted(feat_importances, key=lambda tup: -tup[1]), start=1):
                print(f"#{i}, {lbl}, {importance}")
            print("**-------------------**")

    def update_circle(self):
        super().update_circle()
        if self.circle_item is not None:
            r = self.scaled_radius
            self.circle_item.setRect(QRectF(-r, -r, 2 * r, 2 * r))
            pen = pg.mkPen(QColor(Qt.lightGray), width=1, cosmetic=True)
            self.circle_item.setPen(pen)

    def _add_indicator_item(self, anchor_idx):
        x, y = self.anchor_items[anchor_idx].get_xy()
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self.indicator_item = MoveIndicator(x, y, 600 / dx)
        self.plot_widget.addItem(self.indicator_item)


class InitType(IntEnum):
    Circular, Random = 0, 1

    @staticmethod
    def items():
        return ["Circular", "Random"]


class OWFreeViz(OWAnchorProjectionWidget, ConcurrentWidgetMixin):
    MAX_INSTANCES = 10000

    name = "FreeViz"
    description = "Displays FreeViz projection"
    icon = "icons/Freeviz.svg"
    priority = 240
    keywords = ["viz"]

    settings_version = 3
    initialization = settings.Setting(InitType.Circular)
    GRAPH_CLASS = OWFreeVizGraph
    graph = settings.SettingProvider(OWFreeVizGraph)
    num_neighs = settings.Setting(5)
    sd_factor = settings.Setting(2.0)

    class Inputs:
        data = Input("Data", Table)
        data_subset = Input("Data subset", Table)
        new_data = Input("New data", Table)

    class Error(OWAnchorProjectionWidget.Error):
        no_class_var = widget.Msg("Data has no target variable")
        not_enough_class_vars = widget.Msg(
            "Target variable is not at least binary")
        features_exceeds_instances = widget.Msg(
            "Number of features exceeds the number of instances.")
        too_many_data_instances = widget.Msg("Data is too large.")
        constant_data = widget.Msg("All data columns are constant.")
        not_enough_features = widget.Msg("At least two features are required")

    class Warning(OWAnchorProjectionWidget.Warning):
        removed_features = widget.Msg("Non-binary categorical features are not shown.")

    def __init__(self):
        OWAnchorProjectionWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.new_data = None
        self.new_data_scatter = None
        self.new_data_tooltip = None

    def _add_controls(self):
        self.__add_controls_start_box()
        super()._add_controls()
        self.gui.add_control(
            self._effects_box, gui.hSlider, "Hide radius*:", master=self.graph,
            value="hide_radius", minValue=0, maxValue=100, step=10,
            createLabel=False, callback=self.__radius_slider_changed
        )

        box = gui.widgetBox(self.controlArea, 'Outlier detection settings')
        gui.spin(box, self, value="num_neighs", minv=1, maxv=20, step=1,
                 label="Num. neighbours", alignment=Qt.AlignRight,
                 callback=self.__num_neighs_changed, controlWidth=80)

        gui.spin(box, self, value="sd_factor", minv=0.1, maxv=5.0, step=0.1,
                 label="SD tolerance:", alignment=Qt.AlignRight,
                 callback=self.__tolerance_changed, controlWidth=80, spinType=float)

    def __add_controls_start_box(self):
        box = gui.vBox(self.controlArea, box=True)
        gui.comboBox(
            box, self, "initialization", label="Initialization:",
            items=InitType.items(), orientation=Qt.Horizontal,
            labelWidth=90, callback=self.__init_combo_changed)
        self.run_button = gui.button(box, self, "Start", self._toggle_run)

    @property
    def effective_variables(self):
        return [a for a in self.data.domain.attributes
                if a.is_continuous or a.is_discrete and len(a.values) == 2]

    def __radius_slider_changed(self):
        self.graph.update_radius()

    def __num_neighs_changed(self):
        self.project_new_examples()

    def __tolerance_changed(self):
        self.project_new_examples()

    def __init_combo_changed(self):
        self.Error.proj_error.clear()
        self.init_projection()
        self.setup_plot()
        self.commit()
        if self.task is not None:
            self._run()

    def _toggle_run(self):
        if self.task is not None:
            self.cancel()
            self.graph.set_sample_size(None)
            self.run_button.setText("Resume")
            self.commit()
        else:
            self._run()

    def _run(self):
        if self.data is None:
            return
        self.graph.set_sample_size(self.SAMPLE_SIZE)
        self.run_button.setText("Stop")
        self.start(run_freeviz, self.effective_data, self.projector)

    # ConcurrentWidgetMixin
    def on_partial_result(self, result: Result):
        assert isinstance(result.projector, FreeViz)
        assert isinstance(result.projection, FreeVizModel)
        self.projector = result.projector
        self.projection = result.projection
        self.graph.update_coordinates()
        self.graph.update_density()

    def on_done(self, result: Result):
        assert isinstance(result.projector, FreeViz)
        assert isinstance(result.projection, FreeVizModel)
        self.projector = result.projector
        self.projection = result.projection
        self.graph.set_sample_size(None)
        self.run_button.setText("Start")
        self.commit()

    def on_exception(self, ex: Exception):
        self.Error.proj_error(ex)
        self.graph.set_sample_size(None)
        self.run_button.setText("Start")

    @Inputs.data
    # OWAnchorProjectionWidget
    def set_data(self, data):
        super().set_data(data)
        self.graph.set_sample_size(None)
        if self._invalidated:
            self.init_projection()

    @Inputs.data_subset
    def set_subset_data(self, subset):
        super().set_subset_data(subset)

    @Inputs.new_data
    def set_new_data(self, new_data):
        """ This is intended for projection of new examples, on which the projection is not adjusted. """
        print("Setting new examples!")
        self.new_data = new_data

        # Clear previous projected (additional) examples
        if self.new_data_scatter:
            self.graph.plot_widget.removeItem(self.new_data_scatter)
            self.graph.plot_widget.removeItem(self.new_data_tooltip)
            self.new_data_scatter = None
            self.new_data_tooltip = None

        if self.new_data:
            self.project_new_examples()

    def mark_inliers(self):
        embedded_existing = self.projection(self.data)
        embedded_new = self.projection(self.new_data)
        # FreeViz not initialized
        if not hasattr(self.graph, "feat_importances"):
            print("feat_importances are not defined!")
            return

        # TODO: customizable weights based on how hard it is to tweak a certain feature
        # - use euclidean dissimilarity 1 - 1 / (1 + euclidean dist) for [continuous] features
        # - use hamming distance for [discrete] features
        WEIGHTS = np.ones(len(self.data.domain.attributes)) / len(self.data.domain.attributes)

        # Feature name -> feature importance (= norm of anchor line)
        feat_importances = dict(self.graph.feat_importances)
        feat_importances = [feat_importances[attr.name] for attr in self.data.domain.attributes]

        # No information about actual target variable, can't compare predicted (projected) vs actual
        if not self.new_data.domain.class_var:
            return

        is_inlier = []
        for i in range(len(embedded_new)):
            dists = np.linalg.norm(embedded_new.X[i, :] - embedded_existing.X, axis=1)
            time_reductions = self.new_data.Y[i] - self.data.Y

            nonzero_dists = dists > (0 + 1e-5)
            pos_time_reduction = time_reductions > (0 + 1e-5)

            valid_indices = np.arange(dists.shape[0])[np.logical_and(nonzero_dists, pos_time_reduction)]
            # Sort by: (1) least required changes and (2) biggest time reduction
            recommendation_indices = sorted(valid_indices, key=lambda idx: (dists[idx], -time_reductions[idx]))

            # TODO: this is a WIP (works, but bloats up the visualization)
            # for rank, idx_recommended in enumerate(recommendation_indices[:1], start=1):
            #     print(f"#{rank} distance: {dists[idx_recommended]}, time reduction: {time_reductions[idx_recommended]}")
            #     print(self.data[idx_recommended])
            #     kwargs = dict(x=[embedded_existing.X[idx_recommended, 0] / self.max_insample_norm],
            #                   y=[embedded_existing.X[idx_recommended, 1] / self.max_insample_norm], data=["bla"])
            #     new_point = pg.ScatterPlotItem(**kwargs)
            #     new_point.setSymbol(symbol="+")
            #     new_point.setSize(size=24)
            #     self.graph.plot_widget.addItem(new_point)

            #     new_text = pg.TextItem(text=f"-{time_reductions[idx_recommended]:.2f}",
            #                            anchor=(1, 1), color=(0, 0, 102))
            #     new_text.setPos(embedded_existing.X[idx_recommended, 0] / self.max_insample_norm,
            #                     embedded_existing.X[idx_recommended, 1] / self.max_insample_norm)
            #     self.graph.plot_widget.addItem(new_text)

            closest_indices = np.argsort(dists)[:self.num_neighs]
            closest_examples = embedded_existing[closest_indices]

            mean_target = np.mean(closest_examples.Y)
            sd_target = np.std(closest_examples.Y)
            print("Mean - SD")
            print(f"{mean_target} - {sd_target}")
            print("Actual: ")
            print(embedded_new.Y[i])

            if (mean_target - self.sd_factor * sd_target) <= embedded_new.Y[i] <= (mean_target + self.sd_factor * sd_target):
                is_inlier.append(True)
            else:
                is_inlier.append(False)

        return is_inlier

    def _tooltip_new_points(self, pos):
        """ A hacked together method to display tooltips for new points"""
        act_pos = self.new_data_scatter.mapFromScene(pos)
        found_pts = self.new_data_scatter.pointsAt(act_pos)
        if len(found_pts) != 0:
            tooltip_data = []
            for pt in found_pts:
                tooltip_data.append(pt.data())

            QToolTip.showText(QCursor.pos(), "\n".join(tooltip_data))

    def project_new_examples(self):
        print("Projecting new examples")
        if self.new_data and self.projection:
            new_embedded = self.projection(self.new_data).X / self.max_insample_norm

            inlier_mask = self.mark_inliers()
            pens, brushes = [], []
            for is_inlier in inlier_mask:
                if is_inlier:
                    pens.append(pg.mkPen({'color': '#00400b'}))
                    brushes.append(pg.mkBrush(color='#00ba20'))
                else:
                    pens.append(pg.mkPen({'color': '#400000'}))
                    brushes.append(pg.mkBrush(color='#ba0900'))

            try:
                # Hacked together tooltip data (Meta attributes, followed by attribute values, 1 per line
                formatted_examples = []
                for idx_row in range(len(self.new_data)):
                    metas_formatted = []
                    for idx_meta, curr_meta in enumerate(self.new_data.domain.metas):
                        metas_formatted.append(f"{curr_meta.name}={str(self.new_data.metas[idx_row, idx_meta])}")

                    attrs_formatted = []
                    for idx_attr, curr_attr in enumerate(self.new_data.domain.attributes):
                        value = "?"
                        if isinstance(curr_attr, ContinuousVariable):
                            value = self.new_data.X[idx_row, idx_attr]
                        elif isinstance(curr_attr, DiscreteVariable):
                            value = curr_attr.values[int(self.new_data.X[idx_row, idx_attr])]

                        attrs_formatted.append(f"{curr_attr.name} = {value}")

                    attrs_formatted.append(f"{self.new_data.domain.class_var} = {self.new_data.Y[idx_row]}")

                    formatted_examples.append("<b>[{}]</b>\n{}".format(
                        ','.join(metas_formatted), '\n'.join(attrs_formatted)))

                # Create square symbols for new examples so they can be spotted faster
                # pen_data, brush_data = self.graph.get_colors()
                kwargs = dict(x=new_embedded[:, 0], y=new_embedded[:, 1], data=formatted_examples)
                new_pts = pg.ScatterPlotItem(**kwargs)
                new_pts.setSymbol(symbol="s")
                new_pts.setSize(size=24)
                new_pts.setPen(pens, update=False, mask=None)  # outline
                new_pts.setBrush(brushes, mask=None)  # fill

                # Hacked together tooltips
                self.new_data_scatter = new_pts
                self.new_data_tooltip = pg.TextItem(text='', color=(176, 23, 31), anchor=(1, 1))
                self.graph.plot_widget.addItem(self.new_data_scatter)
                self.graph.plot_widget.addItem(self.new_data_tooltip)
                self.new_data_scatter.scene().sigMouseMoved.connect(self._tooltip_new_points)
            except Exception as e:
                print("NOPE")
                print(e)

    def init_projection(self):
        if self.data is None:
            return
        anchors = FreeViz.init_radial(len(self.effective_variables)) \
            if self.initialization == InitType.Circular \
            else FreeViz.init_random(len(self.effective_variables), 2)
        self.projector = FreeViz(scale=False, center=False,
                                 initial=anchors, maxiter=10)
        data = self.projector.preprocess(self.effective_data)
        self.projector.domain = data.domain
        self.projector.components_ = anchors.T
        self.projection = FreeVizModel(self.projector, self.projector.domain, 2)
        self.projection.pre_domain = data.domain
        self.projection.name = self.projector.name

    def check_data(self):
        def error(err):
            err()
            self.data = None

        super().check_data()
        if self.data is not None:
            class_var, domain = self.data.domain.class_var, self.data.domain
            if class_var is None:
                error(self.Error.no_class_var)
            elif class_var.is_discrete and len(np.unique(self.data.Y)) < 2:
                error(self.Error.not_enough_class_vars)
            elif len(self.data.domain.attributes) < 2:
                error(self.Error.not_enough_features)
            elif len(self.data.domain.attributes) > self.data.X.shape[0]:
                error(self.Error.features_exceeds_instances)
            elif not np.sum(np.std(self.data.X, axis=0)):
                error(self.Error.constant_data)
            elif np.sum(np.all(np.isfinite(self.data.X), axis=1)) > self.MAX_INSTANCES:
                error(self.Error.too_many_data_instances)
            else:
                if len(self.effective_variables) < len(domain.attributes):
                    self.Warning.removed_features()

    def enable_controls(self):
        super().enable_controls()
        self.run_button.setEnabled(self.data is not None)
        self.run_button.setText("Start")

    def get_coordinates_data(self):
        embedding = self.get_embedding()
        if embedding is None:
            return None, None
        valid_emb = embedding[self.valid_data]
        self.max_insample_norm = np.max(np.linalg.norm(valid_emb, axis=1)) or 1
        return valid_emb.T / self.max_insample_norm

    def _manual_move(self, anchor_idx, x, y):
        self.projector.initial[anchor_idx] = [x, y]
        super()._manual_move(anchor_idx, x, y)

    def clear(self):
        super().clear()
        self.cancel()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    @classmethod
    def migrate_settings(cls, _settings, version):
        if version < 3:
            if "radius" in _settings:
                _settings["graph"]["hide_radius"] = _settings["radius"]

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


class MoveIndicator(pg.GraphicsObject):
    def __init__(self, x, y, scene_size, parent=None):
        super().__init__(parent)
        self.arrows = [
            pg.ArrowItem(pos=(x - scene_size * 0.07 * np.cos(np.radians(ang)),
                              y + scene_size * 0.07 * np.sin(np.radians(ang))),
                         parent=self, angle=ang,
                         headLen=13, tipAngle=45,
                         brush=pg.mkColor(128, 128, 128))
            for ang in (0, 90, 180, 270)]

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()


if __name__ == "__main__":  # pragma: no cover
    table = Table("zoo")
    WidgetPreview(OWFreeViz).run(set_data=table, set_new_data=table[::10])

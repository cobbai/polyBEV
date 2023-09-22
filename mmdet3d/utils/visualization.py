import os
import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
import torch.nn.functional as F
from mmcv.runner.dist_utils import master_only


if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg


def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """  # noqa: E501
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')


def convert_overlay_heatmap(feat_map: Union[np.ndarray, torch.Tensor],
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


def wait_continue(figure, timeout: float = 0, continue_key: str = ' ') -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent
    is_inline = 'inline' in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:

        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [
            figure.canvas.mpl_connect(name, handler)  # type: ignore
            for name in ('key_press_event', 'close_event')
        ]

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:
            return 0  # Quit for continue.


class Visualizer:
    def __init__(self, image: Optional[np.ndarray] = None) -> None:
        self.fig_show_cfg = dict(frameon=False)

    def _init_manager(self, win_name: str) -> None:
            """Initialize the matplot manager.

            Args:
                win_name (str): The window name.
            """
            from matplotlib.figure import Figure
            from matplotlib.pyplot import new_figure_manager
            if getattr(self, 'manager', None) is None:
                self.manager = new_figure_manager(
                    num=1, FigureClass=Figure, **self.fig_show_cfg)

            try:
                self.manager.set_window_title(win_name)
            except Exception:
                self.manager = new_figure_manager(
                    num=1, FigureClass=Figure, **self.fig_show_cfg)
                self.manager.set_window_title(win_name)

    @staticmethod
    @master_only
    def draw_featmap(featmap: torch.Tensor,  # 输入格式要求为 CHW
                    overlaid_image: Optional[np.ndarray] = None,  # 果同时输入了 image 数据，则特征图会叠加到 image 上绘制
                    channel_reduction: Optional[str] = 'squeeze_mean',  # 多个通道压缩为单通道的策略
                    topk: int = 20,  # 可选择激活度最高的 topk 个特征图显示
                    arrangement: Tuple[int, int] = (4, 5),  # 多通道展开为多张图时候布局
                    resize_shape: Optional[tuple] = None,  # 可以指定 resize_shape 参数来缩放特征图
                    alpha: float = 0.5  # 图片和特征图绘制的叠加比例
                ) -> np.ndarray:
        """
        输入的 Tensor 一般是包括多个通道的，channel_reduction 参数可以将多个通道压缩为单通道，然后和图片进行叠加显示
            squeeze_mean 将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1, H, W)
            select_max 从输入的 C 维度中先在空间维度 sum，维度变成 (C, )，然后选择值最大的通道
            None 表示不需要压缩，此时可以通过 topk 参数可选择激活度最高的 topk 个特征图显示
        在 channel_reduction 参数为 None 的情况下，topk 参数生效，其会按照激活度排序选择 topk 个通道，然后和图片进行叠加显示，并且此时会通过 arrangement 参数指定显示的布局
            如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示
            如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 channel_reduction来压缩通道。
        考虑到输入的特征图通常非常小，函数支持输入 resize_shape 参数，方便将特征图进行上采样后进行可视化。
        """

        assert isinstance(featmap,
                            torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                            f' but got {type(featmap)}')
        assert featmap.ndim == 3, f'Input dimension must be 3, 'f'but got {featmap.ndim}'
        featmap = featmap.detach().cpu()

        if overlaid_image is not None:
            if overlaid_image.ndim == 2:
                overlaid_image = cv2.cvtColor(overlaid_image,
                                                cv2.COLOR_GRAY2RGB)

            if overlaid_image.shape[:2] != featmap.shape[1:]:
                warnings.warn(
                    f'Since the spatial dimensions of '
                    f'overlaid_image: {overlaid_image.shape[:2]} and '
                    f'featmap: {featmap.shape[1:]} are not same, '
                    f'the feature map will be interpolated. '
                    f'This may cause mismatch problems ！')
                if resize_shape is None:
                    featmap = F.interpolate(
                        featmap[None],
                        overlaid_image.shape[:2],
                        mode='bilinear',
                        align_corners=False)[0]

        if resize_shape is not None:
            featmap = F.interpolate(
                featmap[None],
                resize_shape,
                mode='bilinear',
                align_corners=False)[0]
            if overlaid_image is not None:
                overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max'], \
                f'Mode only support "squeeze_mean", "select_max", ' \
                f'but got {channel_reduction}'
            if channel_reduction == 'select_max':
                sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
                _, indices = torch.topk(sum_channel_featmap, 1)
                feat_map = featmap[indices]
            else:
                feat_map = torch.mean(featmap, dim=0)
            return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
        elif topk <= 0:
            featmap_channel = featmap.shape[0]
            assert featmap_channel in [
                1, 3
            ], ('The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                f'dimension you input is {featmap_channel}, you can use the'
                ' channel_reduction parameter or set topk greater than '
                '0 to solve the error')
            return convert_overlay_heatmap(featmap, overlaid_image, alpha)
        else:
            row, col = arrangement
            channel, height, width = featmap.shape
            assert row * col >= topk, 'The product of row and col in ' \
                                        'the `arrangement` is less than ' \
                                        'topk, please set the ' \
                                        '`arrangement` correctly'

            # Extract the feature map of topk
            topk = min(channel, topk)
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, topk)
            topk_featmap = featmap[indices]

            fig = plt.figure(frameon=False)
            # Set the window layout
            fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            dpi = fig.get_dpi()
            fig.set_size_inches((width * col + 1e-2) / dpi,
                                (height * row + 1e-2) / dpi)
            for i in range(topk):
                axes = fig.add_subplot(row, col, i + 1)
                axes.axis('off')
                axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
                axes.imshow(
                    convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                            alpha))
            image = img_from_canvas(fig.canvas)
            plt.close(fig)
            return image
        
    @master_only
    def show(self,
            drawn_img: Optional[np.ndarray] = None,
            win_name: str = 'image',
            wait_time: float = 0.,
            continue_key: str = ' ',
            backend: str = 'matplotlib') -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. `New in version 0.7.3.`
        """
        if backend == 'matplotlib':
            is_inline = 'inline' in plt.get_backend()
            img = drawn_img  # self.get_image() if  is None else drawn_img
            self._init_manager(win_name)
            fig = self.manager.canvas.figure
            # remove white edges by set subplot margin
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.clear()
            ax = fig.add_subplot()
            ax.axis(False)
            ax.imshow(img)
            self.manager.canvas.draw()

            # Find a better way for inline to show the image
            if is_inline:
                return fig
            wait_continue(fig, timeout=wait_time, continue_key=continue_key)
        elif backend == 'cv2':
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            cv2.namedWindow(winname=f'{id(self)}')
            cv2.setWindowTitle(f'{id(self)}', win_name)
            cv2.imshow(
                str(id(self)),
                drawn_img  # self.get_image() if drawn_img is None else drawn_img
                )
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        elif backend == 'write':
            cv2.imwrite("./tmp/" + str(win_name) + ".jpg", drawn_img)
        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                                f'but got {backend} instead')
        
    def __call__(self, image, win_name, backend="write", channel_reduction="select_max", resize_shape=None) -> Any:
        draw_image = self.draw_featmap(image, channel_reduction=channel_reduction, resize_shape=resize_shape)
        if not os.path.exists("./tmp"): os.mkdir("./tmp")
        self.show(draw_image, win_name=win_name, backend=backend)

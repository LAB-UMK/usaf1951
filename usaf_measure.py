import numpy as np; ar = np.array
import matplotlib.pyplot as plt, matplotlib
from skimage import io, measure, transform, morphology, exposure
from sklearn import cluster
import cv2
from scipy import signal
import sklearn
from scipy.signal import savgol_filter


## Width of 1 line in micrometers in USAF Resolving Power Test Target 1951
## source: https://en.wikipedia.org/wiki/1951_USAF_resolution_test_chart
W = np.array(
    [[2000.00, 1000.00, 500.00, 250.00, 125.00, 62.50, 31.25, 15.63, 7.81, 3.91, 1.95, 0.98],\
     [1781.80, 890.90,  445.45, 222.72, 111.36, 55.68, 27.84, 13.92, 6.96, 3.48, 1.74, 0.87],\
     [1587.40, 793.70,  396.85, 198.43, 99.21,  49.61, 24.80, 12.40, 6.20, 3.10, 1.55, 0.78],\
     [1414.21, 707.11,  353.55, 176.78, 88.39,  44.19, 22.10, 11.05, 5.52, 2.76, 1.38, 0.69],\
     [1259.92, 629.96,  314.98, 157.49, 78.75,  39.37, 19.69, 9.84,  4.92, 2.46, 1.23, 0.62],\
     [1122.46, 561.23,  280.62, 140.31, 70.15,  35.08, 17.54, 8.77,  4.38, 2.19, 1.10, 0.55]])

W = dict(zip([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], W.T))

def _bbox2coords(bbox):
    (min_row, min_col, max_row, max_col) = bbox
    return  ar([[min_row, min_col], [min_row, max_col], \
                [max_row, min_col], [max_row, max_col]] )

def _find_two_rectangles(I):
    R = measure.label(I)
    regions = measure.regionprops(R)
    aspect = [r.major_axis_length / r.minor_axis_length for r in regions if r.minor_axis_length!=0]
    r1, r2 = np.argsort(aspect)[:2]
    r1, r2 = regions[r1], regions[r2]

    if r1.area < r2.area: r1, r2 = r2, r1

    return np.vstack([_bbox2coords(r1.bbox), _bbox2coords(r2.bbox)])

def find_initial_transform_by_rectangles(I, T, plot=True):
    Tb = (T != [255,255,255]).all(axis=2)
    r1 = _find_two_rectangles(I)
    r2 = _find_two_rectangles(Tb)
    tr = transform.estimate_transform('affine', r2[:,::-1], r1[:,::-1])

    if plot:
        r1r = tr.inverse(r1[:,::-1])
        inputr = transform.warp(I, tr, output_shape = Tb.shape)
        fig, ax = plt.subplots()
        ax.imshow(Tb, alpha = 0.5)
        ax.imshow(inputr, alpha = 0.5)
        ax.scatter(r1r.T[0],r1r.T[1])
        ax.scatter(r2.T[1],r2.T[0])
        fig.suptitle("Input-to-template rough transform")

    return np.linalg.inv(tr._inv_matrix)[:2]

def probe_initial_warp_matrix(I, T, sx=1.0, sy=1.0, tx=0.0, ty=0.0, threshold = 0.6):
    Ib = I > threshold
    Tb = T.sum(axis=2) > 0

    warp_matrix = np.array([
            [sx, 0,  tx],
            [0, sy,  ty]], dtype=np.float32)

    Tbr = cv2.warpAffine(Tb.astype(np.float32), warp_matrix, dsize=tuple(np.array(Ib.shape)[::-1]))

    fig, ax = plt.subplots()
    ax.imshow(Ib , alpha=0.5)
    ax.imshow(Tbr, alpha=0.5)
    plt.title('input-to-template warp')
    plt.show()
    return warp_matrix

def warp_coeffs_to_template(I, T, plot=True, initial_warp_matrix = None):
    Ib = I
    Tb = (T != [255,255,255]).all(axis=2)

    if initial_warp_matrix is None:
        convex = morphology.convex_hull_image(morphology.binary_erosion(Ib))
        # convex = morphology.convex_hull_image(Ib)
        bbox = measure.regionprops(convex.astype(int))[0].bbox

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        initial_warp_matrix = np.array([
                [h / T.shape[1], 0,  bbox[1]],
                [0, w / T.shape[0],  bbox[0]]], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400,  0.0001)
    ecc, warp_matrix = cv2.findTransformECC(
        Tb.astype(np.float32), Ib.astype(np.float32),
        initial_warp_matrix.astype(np.float32), cv2.MOTION_AFFINE, criteria, None, 1)

    Tbr = cv2.warpAffine(Tb.astype(np.float32), warp_matrix, dsize=tuple(np.array(Ib.shape)[::-1]))

    if plot:
        fig, ax = plt.subplots()
        ax.cla()
        ax.imshow(Ib , alpha=0.5)
        ax.imshow(Tbr, alpha=0.5)
        plt.title('input-to-template warp')
        plt.show()

    return warp_matrix, Tbr

def measure_resolution(I, warp_matrix, T, plot=True,
                        min_measure_group = -2,
                        max_measure_group = 7):
    Ib = I
    R = measure.label(Ib)

    regions = measure.regionprops(R)
    regions = [r for r in regions if r.area > 2]
    centrs  = [ri.centroid for ri in regions]
    centrs  = np.array(centrs)

    M = np.linalg.inv(np.vstack([warp_matrix, [0,0,1]]))
    centrs_r = M[:2,:2].dot(centrs[:,::-1].T).T + M[:2, 2]
    centrs_r[((centrs_r < 0).any(axis=1))] = [0,0]
    XX, YY = np.round(centrs_r).astype(int).T
    c = T[YY, XX][:,:3]
    # c[:,1] += max_visible_group

    c_uniq, counts = np.unique(c, axis=0, return_counts=True)
    A = c_uniq[counts==3].tolist()
    found = [ci.tolist() in A for ci in c]

    horiz_res = []
    vert_res  = []
    # c = [orient, group, element]
    for ci in c_uniq[counts==3]:
        if ci[1] < min_measure_group: continue
        if ci[1] > max_measure_group: continue
        idxs = [(cii == ci).all() for cii in c]
        xyz = centrs[idxs]
        x,y,z = xyz
        n = np.linalg.norm([x-y, x-z, y-z], axis=1)
        n[n==max(n)] = max(n)/2

        scale = (np.mean(n)/2)/ W[ci[1]][ci[2]-1]

        if ci[0] == 1:
            vert_res.append(scale)
        else:
            horiz_res.append(scale)

    if plot:
        fig, ax_main = plt.subplots()
        plt.tight_layout()
        ax_main.imshow(Ib)
        ax_main.scatter(centrs[:,1], centrs[:,0], c='w', marker='+')
        ax_main.scatter(centrs[found,1], centrs[found,0], c='g', marker='+')

        for i, bbox in enumerate([ri.bbox for ri in regions]):
            if not found[i]: continue
            orient, group, element = c[i]

            if c[i][1] < min_measure_group: continue
            if c[i][1] > max_measure_group: continue
            r = matplotlib.patches.Rectangle([bbox[1], bbox[0]], bbox[3] - bbox[1], bbox[2] - bbox[0], alpha = 0.6)
            ax_main.text(bbox[1], bbox[0], str(group) + ":" + str(element), c='blue')
            ax_main.add_patch(r)

        plt.show()

    return horiz_res, vert_res

def get_group_element(T, M, coords):
    XX, YY     = (T[:,:,:] == coords).all(axis=2).nonzero()
    if len(XX) == 0 or len(YY) == 0 : raise Exception()
    bbox = YY.min(), YY.max(), XX.min(), XX.max()
    p0 = M.dot([bbox[0], bbox[2], 1])[:2]
    p1 = M.dot([bbox[1], bbox[3], 1])[:2]
    return p0, p1

def flat_argrelextrema(serie, order, comparator = np.greater_equal):
    idxs = signal.argrelextrema(serie, comparator, order = order)[0]
    if len(idxs) < 2: return np.sort(idxs)

    a = cluster.AgglomerativeClustering(
                n_clusters = None,
                distance_threshold = order).fit(idxs.reshape(-1,1))

    idxs = [np.mean(idxs[a.labels_ == l]).astype(int) for l in np.unique(a.labels_)]
    return np.sort(idxs)

def MTF(input, warp_matrix, template, margin = 0.0, plot=True, max_visible_group = 0):
    # template = template + [0, 0, max_visible_group]
    M = np.vstack([warp_matrix, [0,0,1]])
    ax_contrasts_v, ax_contrasts_h = None, None

    if plot:
        fig, ax = plt.subplots()
        plt.tight_layout()

        fig2, (ax_contrasts_h, ax_contrasts_v, ax_resolution) = plt.subplots(nrows=3)
        plt.tight_layout()
        ax.imshow(input)
        ax_resolution.set_xlabel('line pair for mm')
        ax_contrasts_h.set_ylabel('contrast')
        ax_contrasts_v.set_ylabel('contrast')
        ax_resolution.set_ylabel('contrast')

    contrasts_lp_h, contrasts_lp_v = [], []
    lppermm_v, lppermm_h     = [], []
    for orient, fc, contrasts_lp, lppermm in zip([1,2], 'bg', [contrasts_lp_v, contrasts_lp_h], [lppermm_v, lppermm_h]):
        for group in [4, 5, 6]:
            for el in range(1,7):
                try:
                    p0, p1 = get_group_element(template, M, [orient, group, el])
                except:
                    print ('group,element %s not found!' % ([group, el]))
                    continue
                coords = [orient, group, el]
                T = template
                XX, YY     = (T[:,:,:] == coords).all(axis=2).nonzero()
                w, h = p1 - p0
                p0[0] += margin*w
                p1[0] -= margin*w

                p0[1] += margin *h
                p1[1] -= margin *h

                w, h = p1 - p0

                if plot:
                    rect = matplotlib.patches.Rectangle(p0, width = w, height = h, alpha=0.5, fc=fc )
                    ax.add_patch(rect)

                p0, p1 = np.floor(p0).astype(int), np.ceil(p1).astype(int)
                Igrel = input[p0[1]:p1[1], p0[0]:p1[0]]
                contrasts_lp.append(Igrel.mean(axis=1 if orient == 2 else 0))
                lppermm.append(2**(group + (el-1)/6 ))

    contrasts_v, contrasts_h = [], []
    for ret_contrasts, contrasts_lp, ax_contrasts in \
            zip([contrasts_v, contrasts_h],
                [contrasts_lp_v, contrasts_lp_h],
                [ax_contrasts_v, ax_contrasts_h]):

        ss = [0] + list(map(len, contrasts_lp))
        ss = np.cumsum(ss)
        contrasts_lp = np.hstack(contrasts_lp)
        contrasts_lp -= contrasts_lp.min()
        contrasts_lp /= contrasts_lp.max()

        for i, (xs, xe) in enumerate(zip(ss[:-1], ss[1:])):
            c_slice = contrasts_lp[xs:xe]
            max_idxs = flat_argrelextrema(c_slice, (xe-xs)//5, np.greater_equal)
            min_idxs = flat_argrelextrema(c_slice, (xe-xs)//5, np.less_equal)
            if max_idxs == [] or min_idxs == []: print ('dupa')
            if (min_idxs[0] == 0): min_idxs = min_idxs[1:]
            if (max_idxs[0] == 0): max_idxs = max_idxs[1:]

            if (min_idxs[-1] == len(c_slice)-1): min_idxs = min_idxs[:-1]
            if (max_idxs[-1] == len(c_slice)-1): max_idxs = max_idxs[:-1]

            ret_contrasts.append(c_slice[max_idxs].mean() - c_slice[min_idxs].mean())

            if plot:
                ax_contrasts.plot(np.arange(xs, xe), c_slice)
                ax_contrasts.plot(np.arange(xs, xe)[max_idxs], c_slice[max_idxs], 'go')
                ax_contrasts.plot(np.arange(xs, xe)[min_idxs], c_slice[min_idxs], 'ro')

                ax_contrasts.plot([xs, xe], [c_slice[max_idxs].mean(), c_slice[max_idxs].mean()] )
                ax_contrasts.plot([xs, xe], [c_slice[min_idxs].mean(), c_slice[min_idxs].mean()] )

    for ax_contrasts in [ax_contrasts_v, ax_contrasts_h]:
        ax_contrasts.axhline(0, c = 'r')
        ax_contrasts.axhline(1, c = 'r')
        ax_contrasts.set_ylim(-0.05, 1.05)

    lppermm, contrasts_v, contrasts_h = ar(lppermm), ar(contrasts_v), ar(contrasts_h)
    # lppermm, group_el_contrasts = np.array(lppermm), np.array(group_el_contrasts)

    if plot:
        ax_resolution.plot(lppermm, contrasts_v, c='b', label ='vertical')
        ax_resolution.plot(lppermm, contrasts_h, c='g', label ='horizontal')

        a, b, c, d = np.polyfit(lppermm, contrasts_v, 3)
        ax_resolution.plot(lppermm, a*lppermm**3 + b*lppermm**2 + c*lppermm + d, 'b--')

        a, b, c, d = np.polyfit(lppermm, contrasts_h, 3)
        ax_resolution.plot(lppermm, a*lppermm**3 + b*lppermm**2 + c*lppermm + d, 'g--')

        ax_resolution.legend()
        ax_resolution.grid()

    return lppermm, np.vstack([contrasts_v, contrasts_h])

def MTF_local_offset(input, warp_matrix, template, margin = 0.0, plot=True):
    M = np.vstack([warp_matrix, [0,0,1]])
    ax_contrasts_v, ax_contrasts_h = None, None

    if plot:
        fig, ax = plt.subplots()
        plt.tight_layout()
        fig2, (ax_contrasts_h, ax_contrasts_v, ax_resolution) = plt.subplots(nrows=3)
        plt.tight_layout()
        ax.imshow(input)


    contrasts_lp_h, contrasts_lp_v = [], []
    lppermm_v, lppermm_h     = [], []
    for orient, fc, contrasts_lp, lppermm in zip([0,1], 'bg',
                            [contrasts_lp_v, contrasts_lp_h], [lppermm_v, lppermm_h]):
        for group in [4,5,6]:
            for el in range(1,7):
                p0, p1 = get_group_element(template, M, [orient, group, el, 255])
                w, h = p1 - p0
                if orient == 0:
                    p0[0] -= margin*w
                    p1[0] += margin*w

                    p0[1] += margin *h
                    p1[1] -= margin *h
                else :
                    p0[0] += margin*w
                    p1[0] -= margin*w

                    p0[1] -= margin *h
                    p1[1] += margin *h

                w, h = p1 - p0

                if plot:
                    rect = matplotlib.patches.Rectangle(p0, width = w, height = h, alpha=0.5, fc=fc )
                    ax.add_patch(rect)

                p0, p1 = np.floor(p0).astype(int), np.ceil(p1).astype(int)
                Igrel = input[p0[1]:p1[1], p0[0]:p1[0]]
                # contrasts_lp.append(Igrel.mean(axis=1 if orient == 1 else 0))
                contrasts_lp.append(np.nanmean(Igrel, axis=1 if orient == 1 else 0))
                lppermm.append(2**(group + (el-1)/6 ))

    contrasts_v, contrasts_h = [], []
    for ret_contrasts, contrasts_lp, ax_contrasts in \
            zip([contrasts_v, contrasts_h],
                [contrasts_lp_v, contrasts_lp_h],
                [ax_contrasts_v, ax_contrasts_h]):

        ss = [0] + list(map(len, contrasts_lp))
        ss = np.cumsum(ss)
        contrasts_lp = np.hstack(contrasts_lp)
        # contrasts_lp -= contrasts_lp.min()
        # contrasts_lp /= contrasts_lp.max()
        for i, (xs, xe) in enumerate(zip(ss[:-1], ss[1:])):
            c_slice = contrasts_lp[xs:xe]
            c_slice -= c_slice.min()
            max_idxs = flat_argrelextrema(c_slice, (xe-xs)//5, np.greater_equal)
            min_idxs = flat_argrelextrema(c_slice, (xe-xs)//5, np.less_equal)

            # if (min_idxs[0] == 0): min_idxs = min_idxs[1:]
            # if (max_idxs[0] == 0): max_idxs = max_idxs[1:]
            #
            # if (min_idxs[-1] == len(c_slice)-1): min_idxs = min_idxs[:-1]
            # if (max_idxs[-1] == len(c_slice)-1): max_idxs = max_idxs[:-1]

            # ret_contrasts.append(
            #     (c_slice[max_idxs].mean() - c_slice[min_idxs[1:-1]].mean()) /
            #     (c_slice[max_idxs].mean()) )

            ret_contrasts.append(
                (c_slice[max_idxs].mean() - c_slice[min_idxs[1:-1]].mean() ) )

            if plot:
                ax_contrasts.plot(np.arange(xs, xe), c_slice)
                ax_contrasts.plot(np.arange(xs, xe)[max_idxs], c_slice[max_idxs], 'go')
                ax_contrasts.plot(np.arange(xs, xe)[min_idxs], c_slice[min_idxs], 'ro')

                ax_contrasts.plot([xs, xe], [c_slice[max_idxs].mean(), c_slice[max_idxs].mean()] )
                ax_contrasts.plot([xs, xe], [c_slice[min_idxs[1:-1]].mean(), c_slice[min_idxs[1:-1]].mean()] )

    for ax_contrasts in [ax_contrasts_v, ax_contrasts_h]:
        ax_contrasts.axhline(0, c = 'r')
        ax_contrasts.axhline(1, c = 'r')
        ax_contrasts.set_ylim(-0.05, 1.05)


    lppermm, contrasts_v, contrasts_h = ar(lppermm), ar(contrasts_v), ar(contrasts_h)
    # lppermm, group_el_contrasts = np.array(lppermm), np.array(group_el_contrasts)

    if plot:
        ax_resolution.plot(lppermm, contrasts_v, c='b', label ='vertical')
        ax_resolution.plot(lppermm, contrasts_h, c='g', label ='horizontal')

        a, b, c, d = np.polyfit(lppermm, contrasts_v, 3)
        ax_resolution.plot(lppermm, a*lppermm**3 + b*lppermm**2 + c*lppermm + d, 'b--')

        a, b, c, d = np.polyfit(lppermm, contrasts_h, 3)
        ax_resolution.plot(lppermm, a*lppermm**3 + b*lppermm**2 + c*lppermm + d, 'g--')

        ax_resolution.legend()
        ax_resolution.grid()

    return lppermm, np.vstack([contrasts_v, contrasts_h])

def redraw_fig(ax):
    fig = ax.get_figure()
    fig.canvas.update()
    fig.canvas.draw()
    fig.canvas.flush_events()

threshold = 0
def preprocess_input(input, plot = True):
    global threshold
    input = exposure.rescale_intensity(input, in_range=(np.nanmin(input), np.nanmax(input)), out_range=(0,1))
    input[np.isnan(input)] = np.nanmean(input)
    freqs, bins = exposure.histogram(input, nbins=150)
    min_bin = np.argmin(freqs[20:-40])
    threshold = bins[min_bin + 20]
    input_b = input >  threshold

    if plot:
        fig = plt.figure(figsize=(6,6))
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,1), sharex = ax1, sharey = ax1)
        ax3 = plt.subplot2grid((2,2), (1,0), colspan = 2)
        ax3.plot(bins, freqs)
        ax3.plot(bins, savgol_filter(freqs, 11, 2), '--')
        ax3.axvline(threshold, c='r')
        ax3.set_title('Click to change threshold')
        ax1.imshow(input)
        ax2.imshow(input_b)

        def onclick(event):
            global threshold
            if (event.inaxes == ax3):
                threshold = event.xdata
                ax3.cla()
                ax2.cla()
                ax3.plot(bins, freqs)
                ax3.plot(bins, savgol_filter(freqs, 11, 2), '--')
                ax3.axvline(threshold, c='r')
                input_b = input >  threshold
                ax2.imshow(input_b)
                redraw_fig(ax3)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        fig.tight_layout()
        plt.show(block=True)

    return input, input > threshold, threshold

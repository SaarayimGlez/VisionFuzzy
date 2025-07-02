%Esta es una función alterna a la que se usa originalmente, es usada para
%procesar los histogramas que son demasiado grandes y pueden causar out of
%memory. Procesa los conjuntos RGB para obtener el espacio de color HSI

function [train, test, clase, superClase] = structDB2TrainingandTest(db, ini, fin, dimensionType)
    
    isHS = strcmp(dimensionType, '3DRGB2HS');
    isHSI = strcmp(dimensionType, '3DRGB2HSI');
    useHS = isHS || isHSI;


    numSamples = fin - ini + 1;
    if isHS
        activeBinsGlobal = false(1, 256*256);
    elseif isHSI
        activeBinsGlobal = false(1, 256^3);
    else
        activeBinsGlobal = false(1, 256^3);
    end

    clase = cell(numSamples, 1);
    superClase = cell(numSamples, 1);

    for k = ini:fin
        data = db.dataset(k);
        switch dimensionType
            case '3DLab'
                res_train = reshape(data.h3d_lab_train, [], 1)';
                res_test  = reshape(data.h3d_lab_test, [], 1)';
            case '3DRGB'
                res_train = reshape(data.h3d_rgb_train, [], 1)';
                res_test  = reshape(data.h3d_rgb_test, [], 1)';
            case '3DHSI'
                res_train = reshape(data.h3d_hsi_train, [], 1)';
                res_test  = reshape(data.h3d_hsi_test, [], 1)';
            case '3DRGB2HSI'
                res_train = rgbHistToHsiVectorized(data.h3d_rgb_train, false);
                res_test  = rgbHistToHsiVectorized(data.h3d_rgb_test, false);
            case '3DRGB2HS'
                res_train = rgbHistToHsiVectorized(data.h3d_rgb_train, true);
                res_test  = rgbHistToHsiVectorized(data.h3d_rgb_test, true);
            otherwise
                error('Tipo de dimension no soportado: %s', dimensionType);
        end

        activeBinsGlobal = activeBinsGlobal | (res_train ~= 0) | (res_test ~= 0);

        clase{k - ini + 1} = data.clase;
        superClase{k - ini + 1} = data.superClase;
    end

    idxBins = find(activeBinsGlobal);
    numActiveBins = numel(idxBins);

    train = zeros(numSamples, numActiveBins, 'single');
    test  = zeros(numSamples, numActiveBins, 'single');

    for k = ini:fin
        data = db.dataset(k);
        switch dimensionType
            case '3DLab'
                res_train = reshape(data.h3d_lab_train, [], 1)';
                res_test  = reshape(data.h3d_lab_test, [], 1)';
            case '3DRGB'
                res_train = reshape(data.h3d_rgb_train, [], 1)';
                res_test  = reshape(data.h3d_rgb_test, [], 1)';
            case '3DHSI'
                res_train = reshape(data.h3d_hsi_train, [], 1)';
                res_test  = reshape(data.h3d_hsi_test, [], 1)';
            case '3DRGB2HSI'
                res_train = rgbHistToHsiVectorized(data.h3d_rgb_train, false);
                res_test  = rgbHistToHsiVectorized(data.h3d_rgb_test, false);
            case '3DRGB2HS'
                res_train = rgbHistToHsiVectorized(data.h3d_rgb_train, true);
                res_test  = rgbHistToHsiVectorized(data.h3d_rgb_test, true);
        end

        train(k - ini + 1, :) = single(res_train(idxBins));
        test(k - ini + 1, :)  = single(res_test(idxBins));
    end
end


function vec = rgbHistToHsiVectorized(rgb_hist, onlyHS)


    rgb_hist = double(rgb_hist);

    [R, G, B] = ind2sub(size(rgb_hist), find(rgb_hist > 0));
    vals = rgb_hist(rgb_hist > 0);

    r = (R - 1) / 255;
    g = (G - 1) / 255;
    b = (B - 1) / 255;

    i = (r + g + b) / 3;

    minRGB = min([r, g, b], [], 2);
    s = 1 - minRGB ./ (i + eps);

    num = 0.5 * ((r - g) + (r - b));
    den = sqrt((r - g).^2 + (r - b) .* (g - b)) + eps;
    theta = acos(num ./ den);
    h = theta;
    h(b > g) = 2*pi - h(b > g);
    h = h / (2*pi);
    h(s < 1e-6) = 0;

    h_bin = uint32(floor(h * 255) + 1);
    s_bin = uint32(floor(s * 255) + 1);
    i_bin = uint32(floor(i * 255) + 1);

    if onlyHS
        idx = sub2ind([256, 256], h_bin, s_bin);
        vec = sparse(1, idx, vals, 1, 256 * 256); 
    else
        idx = sub2ind([256, 256, 256], h_bin, s_bin, i_bin);
        vec = sparse(1, idx, vals, 1, 256^3);
    end
end


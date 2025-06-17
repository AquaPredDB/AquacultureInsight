classdef spatialDropoutLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Example custom spatial dropout layer.

    properties
        DropoutFactor
    end

    methods
        function layer = spatialDropoutLayer(dropoutFactor,NameValueArgs)
            % layer = spatialDropoutLayer creates a spatial dropout layer
            % with dropout factor 0.02;
            %
            % layer = spatialDropoutLayer(dropoutProb) creates a spatial
            % dropout layer with the specified probability.
            %
            % layer = spatialDropoutLayer(__,Name=name) also specifies the
            % layer name using any of the previous syntaxes.

            % Parse input arguments.
            arguments
                dropoutFactor = 0.02;
                NameValueArgs.Name = ""
            end
            name = NameValueArgs.Name;

            % Set layer properties.
            layer.Name = name;
            layer.Description = "Spatial dropout with factor " + dropoutFactor;
            layer.Type = "Spatial Dropout";
            layer.DropoutFactor = dropoutFactor;
        end

        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Output:
            %         Z     - Output of layer forward function

            % At prediction time, the output is unchanged.
            Z = X;
        end

        function Z = forward(layer, X)
            % Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Output:
            %         Z - Output of layer forward function

            dropoutFactor = layer.DropoutFactor;

            % Mask dimensions.
            fmt = dims(X);
            maskSize = size(X);
            maskSize(ismember(fmt,'ST')) = 1;

            % Create mask.
            dropoutScaleFactor = single(1 - dropoutFactor);
            dropoutMask = (rand(maskSize,'like',X) > dropoutFactor) / dropoutScaleFactor;

            % Dropout.
            Z = X .* dropoutMask;
        end
    end
end
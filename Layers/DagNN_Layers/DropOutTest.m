classdef DropOutTest < dagnn.ElementWise
  properties
    rate = 0.5
    frozen = false
  end

  properties (Transient)
    mask
  end

  methods
    function outputs = forward(obj, inputs, params)
        [outputs{1}, obj.mask] = vl_nndropout(inputs{1}, 'rate', obj.rate) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nndropout(inputs{1}, derOutputs{1}, 'mask', obj.mask) ;
      derParams = {} ;
    end

    % ---------------------------------------------------------------------
    function obj = DropOutTest(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      obj.mask = [] ;
      obj.frozen = false ;
    end
  end
end

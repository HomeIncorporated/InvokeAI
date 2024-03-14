import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { socketModelLoadComplete, socketModelLoadStarted } from 'services/events/actions';

const log = logger('socketio');

export const addModelLoadEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketModelLoadStarted,
    effect: (action) => {
      const { config, submodel_type } = action.payload.data;

      let message = `Model load started: ${config.name} (${config.key})`;

      if (submodel_type) {
        message = message.concat(`/${submodel_type}`);
      }

      log.debug(action.payload, message);
    },
  });

  startAppListening({
    actionCreator: socketModelLoadComplete,
    effect: (action) => {
      const { config, submodel_type } = action.payload.data;

      let message = `Model load complete: ${config.name} (${config.key})`;

      if (submodel_type) {
        message = message.concat(`/${submodel_type}`);
      }

      log.debug(action.payload, message);
    },
  });
};

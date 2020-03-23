export interface HelmetDetectionClass {
    name: string;
    id: number;
    displayName: string;
  }
  
  export const CLASSES: {[key: string]: HelmetDetectionClass} = {
    1: {
      name: 'person',
      id: 1,
      displayName: 'person',
    },
    2: {
      name: 'hat',
      id: 2,
      displayName: 'helmet',
    }
  };